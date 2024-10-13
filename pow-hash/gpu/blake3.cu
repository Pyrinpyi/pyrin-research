#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstring>

#define OUT_LEN 32
#define KEY_LEN 32
#define BLOCK_LEN 64
#define CHUNK_LEN 1024
#define CHAINING_VALUE_LEN 8

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define PARENT (1 << 2)
#define ROOT (1 << 3)
#define KEYED_HASH (1 << 4)
#define DERIVE_KEY_CONTEXT (1 << 5)
#define DERIVE_KEY_MATERIAL (1 << 6)

// ---- DEBUG

__device__ void hex_dump(const char* desc, const void* addr, const int len) {
    int i;
    unsigned char buff[17];
    const unsigned char* pc = (const unsigned char*)addr;

    if (desc != NULL)
        printf ("%s (%d bytes):\n", desc, len);

    for (i = 0; i < len; i++) {
        if ((i % 16) == 0) {
            if (i != 0)
                printf("  %s\n", buff);
            printf("  %04x ", i);
        }

        printf(" %02x", pc[i]);

        if ((pc[i] < 0x20) || (pc[i] > 0x7e))
            buff[i % 16] = '.';
        else
            buff[i % 16] = pc[i];
        buff[(i % 16) + 1] = '\0';
    }

    while ((i % 16) != 0) {
        printf("   ");
        i++;
    }

    printf("  %s\n", buff);
}

// ---


__constant__ uint32_t d_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__constant__ uint8_t d_MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

__device__ uint32_t mask32(uint32_t x) {
    return x & 0xFFFFFFFF;
}

__device__ uint32_t add32(uint32_t x, uint32_t y) {
    return mask32(x + y);
}

__device__ uint32_t rightrotate32(uint32_t x, int n) {
    return mask32(x << (32 - n)) | (x >> n);
}

__device__ void g(uint32_t* state, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    state[a] = add32(state[a], add32(state[b], mx));
    state[d] = rightrotate32(state[d] ^ state[a], 16);
    state[c] = add32(state[c], state[d]);
    state[b] = rightrotate32(state[b] ^ state[c], 12);
    state[a] = add32(state[a], add32(state[b], my));
    state[d] = rightrotate32(state[d] ^ state[a], 8);
    state[c] = add32(state[c], state[d]);
    state[b] = rightrotate32(state[b] ^ state[c], 7);
}

__device__ void blake3_round(uint32_t* state, const uint32_t* m) {
    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ void permute(uint32_t* m) {
    uint32_t original[16];
    for (int i = 0; i < 16; i++) {
        original[i] = m[i];
    }
    for (int i = 0; i < 16; i++) {
        m[i] = original[d_MSG_PERMUTATION[i]];
    }
}

__device__ void compress(const uint32_t* chaining_value, const uint32_t* block_words,
                         uint64_t counter, uint32_t block_len, uint32_t flags,
                         uint32_t* out) {
    uint32_t state[16];
    for (int i = 0; i < 8; i++) {
        state[i] = chaining_value[i];
    }
    for (int i = 8; i < 16; i++) {
        state[i] = d_IV[i - 8];
    }
    state[12] = mask32(counter);
    state[13] = mask32(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    uint32_t block[16];
    for (int i = 0; i < 16; i++) {
        block[i] = block_words[i];
    }

    blake3_round(state, block);  // round 1
    permute(block);
    blake3_round(state, block);  // round 2
    permute(block);
    blake3_round(state, block);  // round 3
    permute(block);
    blake3_round(state, block);  // round 4
    permute(block);
    blake3_round(state, block);  // round 5
    permute(block);
    blake3_round(state, block);  // round 6
    permute(block);
    blake3_round(state, block);  // round 7

    for (int i = 0; i < 8; i++) {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }

    for (int i = 0; i < 16; i++) {
        out[i] = state[i];
    }
}

typedef struct {
    uint32_t key[8];
    uint8_t block[BLOCK_LEN];
    uint32_t chaining_value[CHAINING_VALUE_LEN];
    size_t block_len;
    uint64_t counter;
    uint32_t flags;
} Blake3State;

__device__ void chunk_state_init(Blake3State* self, const uint32_t* key, uint32_t flags) {
    memcpy(self->chaining_value, key, CHAINING_VALUE_LEN * sizeof(uint32_t));
    self->counter = 0;
    self->block_len = 0;
    self->flags = flags | CHUNK_START;
}

__device__ void chunk_state_update(Blake3State* self, const uint8_t* input, size_t input_len) {
    uint32_t block_words[16];

    while (input_len > 0) {
        if (self->block_len == BLOCK_LEN) {
            memcpy(block_words, self->block, BLOCK_LEN);
            uint32_t out[16];
            compress(self->chaining_value, block_words, self->counter, BLOCK_LEN, self->flags, out);
            memcpy(self->chaining_value, out, CHAINING_VALUE_LEN * sizeof(uint32_t));
            self->block_len = 0;
            self->counter++;
            self->flags &= ~CHUNK_START;
        }

        size_t take = BLOCK_LEN - self->block_len;
        if (take > input_len) {
            take = input_len;
        }

        memcpy(self->block + self->block_len, input, take);
        self->block_len += take;
        input += take;
        input_len -= take;
    }

    hex_dump("Updated chaining value", self->chaining_value, sizeof(self->chaining_value));
}

__device__ void chunk_state_finalize(const Blake3State* self, uint32_t* out) {
    uint32_t block_words[16];
    memcpy(block_words, self->block, self->block_len);
    memset((uint8_t*)block_words + self->block_len, 0, BLOCK_LEN - self->block_len);

    hex_dump("Before last compress", block_words, self->block_len);
    hex_dump("blcok", self->block, self->block_len);
    hex_dump("chaining_value", self->chaining_value, self->block_len);

    compress(self->chaining_value, block_words, self->counter, self->block_len, self->flags | CHUNK_END, out);

    hex_dump("Finalized output", out, OUT_LEN);
}

// For using inside CUDA
// __global__ void blake3_hash_kernel(const uint8_t* input, uint32_t input_len, uint8_t* output) {
//     Blake3State state;
//     blake3_init_kernel(&state, NULL, 0);
//     chunk_state_update(&state, input, input_len);
//     chunk_state_finalize(&state, output);
// }

extern "C" {
    __global__ void blake3_init_kernel(Blake3State* state, const uint8_t* key, uint32_t flags) {
        uint32_t key_words[8];
        if (flags & KEYED_HASH) {
            memcpy(key_words, key, KEY_LEN);
        } else {
            memcpy(key_words, d_IV, KEY_LEN);
        }
        flags |= CHUNK_END | ROOT; // CHUNK_START is added by init
        hex_dump("blake3_init_kernel key_words", key_words, KEY_LEN);
        chunk_state_init(state, key_words, flags);
    }

    __global__ void blake3_update_kernel(Blake3State* state, const uint8_t* input, size_t input_len) {
        hex_dump("blake3_update_kernel", input, input_len);
        chunk_state_update(state, input, input_len);
    }

    __global__ void blake3_finalize_kernel(Blake3State* state, uint32_t* output) {
        chunk_state_finalize(state, output);
    }

    // __global__ void blake3_hash_kernel(const uint8_t* input, uint32_t input_len, uint8_t* output) {
    //     uint32_t chaining_value[8];

    //     for (int i = 0; i < 8; i++) {
    //         chaining_value[i] = d_IV[i];
    //     }

    //     uint32_t block_words[16] = {0};
    //     for (uint32_t i = 0; i < input_len; i++) {
    //         ((uint8_t*)block_words)[i] = input[i];
    //     }

    //     uint32_t out[16];
    //     compress(chaining_value, block_words, 0, input_len, CHUNK_START | CHUNK_END | ROOT, out);

    //     for (int i = 0; i < OUT_LEN; i++) {
    //         output[i] = ((uint8_t*)out)[i];
    //     }
    // }

    __global__ void blake3_hash_kernel(const uint8_t* input, uint32_t input_len, uint8_t* output) {
        uint32_t chaining_value[8];
        uint32_t block_words[16];
        uint64_t counter = 0;
        uint32_t flags = CHUNK_START;

        // Initialize chaining value with IV
        for (int i = 0; i < 8; i++) {
            chaining_value[i] = d_IV[i];
        }

        // Process full blocks
        while (input_len > BLOCK_LEN) {
            // Copy input block to block_words
            for (int i = 0; i < 16; i++) {
                block_words[i] = ((uint32_t*)input)[i];
            }

            uint32_t out[16];
            compress(chaining_value, block_words, counter, BLOCK_LEN, flags, out);

            // Update chaining value
            for (int i = 0; i < 8; i++) {
                chaining_value[i] = out[i];
            }

            input += BLOCK_LEN;
            input_len -= BLOCK_LEN;
            counter++;
            flags = 0; // Clear CHUNK_START flag after first block
        }

        // Process last block
        memset(block_words, 0, sizeof(block_words));
        memcpy(block_words, input, input_len);

        flags |= CHUNK_END | ROOT;
        uint32_t out[16];
        compress(chaining_value, block_words, counter, input_len, flags, out);

        // Copy output
        for (int i = 0; i < OUT_LEN; i++) {
            output[i] = ((uint8_t*)out)[i];
        }
    }
}