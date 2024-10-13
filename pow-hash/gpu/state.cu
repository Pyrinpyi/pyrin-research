#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>


// ---- DEBUG

__device__ void hex_dump(const char* desc, const void* addr, const int len) {
    int i;
    unsigned char buff[17];
    const unsigned char* pc = (const unsigned char*)addr;

    if (desc != NULL)
        printf ("%s:\n", desc);

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

typedef struct __attribute__((packed)) {
    uint16_t version;
    // uint64_t parents_by_level_count;
    // char** parents_by_level;  // dynamic 2D array parents of parents [["hash"]]
    uint8_t hash_merkle_root[32];
    uint8_t accepted_id_merkle_root[32];
    uint8_t utxo_commitment[32];
    uint64_t timestamp;
    uint32_t bits;
    uint64_t nonce;
    uint64_t daa_score;
    uint64_t blue_score;
    uint64_t blue_work_length;
    uint8_t* blue_work;  // This will be a dynamic array
    char pruning_point[32];
} BlockHeader;

struct BlockHeaderData {
    uint16_t version;
    // uint64_t parents_by_level_count;
    // uint64_t* parents_by_level_size;
    // uint8_t* parents_by_level_data;
    // uint8_t hash_merkle_root[32];
    // uint8_t accepted_id_merkle_root[32];
    // uint8_t utxo_commitment[32];
    uint64_t timestamp;
    uint32_t bits;
    uint64_t nonce;
    uint64_t daa_score;
    // uint64_t blue_score;
    // uint64_t blue_work_length;
    // uint8_t* blue_work;
    // uint8_t pruning_point[32];
};

__global__ void calculate_block_header_hash(BlockHeaderData* data, uint8_t* output_hash) {

    printf("Version: %d\n", data->version);
    printf("Timestamp: %llu\n", data->timestamp);
    printf("Bits: %u\n", data->bits);
    printf("DAA Score: %u\n", data->daa_score);
    printf("\n");

    output_hash[0] = 1;
    output_hash[1] = 2;
    output_hash[2] = 3;
    output_hash[3] = 4;

    // blake3_hasher hasher;
    // uint8_t key[32] = "BlockHash\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    // blake3_hasher_init_keyed(&hasher, key);

    // blake3_hasher_update(&hasher, &data->version, sizeof(uint16_t));
    // blake3_hasher_update(&hasher, &data->parents_by_level_count, sizeof(uint64_t));

    // for (uint64_t i = 0; i < data->parents_by_level_count; i++) {
    //     uint64_t level_length = data->parents_by_level_lengths[i];
    //     blake3_hasher_update(&hasher, &level_length, sizeof(uint64_t));
    //     blake3_hasher_update(&hasher, data->parents_by_level_data + i * 32 * level_length, 32 * level_length);
    // }

    // blake3_hasher_update(&hasher, data->hash_merkle_root, 32);
    // blake3_hasher_update(&hasher, data->accepted_id_merkle_root, 32);
    // blake3_hasher_update(&hasher, data->utxo_commitment, 32);
    // blake3_hasher_update(&hasher, &data->timestamp, sizeof(uint64_t));
    // blake3_hasher_update(&hasher, &data->bits, sizeof(uint32_t));
    // blake3_hasher_update(&hasher, &data->nonce, sizeof(uint64_t));
    // blake3_hasher_update(&hasher, &data->daa_score, sizeof(uint64_t));
    // blake3_hasher_update(&hasher, &data->blue_score, sizeof(uint64_t));
    // blake3_hasher_update(&hasher, &data->blue_work_length, sizeof(uint64_t));
    // blake3_hasher_update(&hasher, data->blue_work, data->blue_work_length);
    // blake3_hasher_update(&hasher, data->pruning_point, 32);

    // blake3_hasher_finalize(&hasher, output_hash, BLAKE3_OUT_LEN);
}

typedef struct {
    bool is_synced;
    BlockHeader header;
} State;

__global__ void update_state(State* state, bool new_value) {
    state->is_synced = new_value;
}

__global__ void get_state(State* state, bool *result) {
    *result = state->is_synced;
}

// __global__ void update_block_header(State* state, BlockHeader* header) {
__global__ void update_block_header(BlockHeader* header) {

    // hex_dump("header", header, sizeof(BlockHeader));

    // BlockHeader* h = &header[blockIdx.x];
    BlockHeader* h = header;
    // state->header = *h;

    printf("Version: %d\n", h->version);
    printf("Timestamp: %llu\n", h->timestamp);
    printf("DAA Score: %llu\n", h->daa_score);

    printf("\n");
    
    /*printf("Hash Merkle Root: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", h->hash_merkle_root[i]);
    }
    printf("\n");
    
    printf("Accepted Id Merkle Root: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", h->accepted_id_merkle_root[i]);
    }
    printf("\n");*/
    
    hex_dump("blue_work", h->blue_work, h->blue_work_length);
    hex_dump("Utxo Commitment", h->utxo_commitment, 32);
    hex_dump("Pruning Point", h->pruning_point, 32);
    
    printf("\n");
}

// __global__ void update_block_header(State* state, BlockHeader* header) {
//     state->header = header;

//     // Print all fields of the BlockHeader
//     printf("Updating BlockHeader:\n");
//     printf("Version: %u\n", header->version);
//     /*printf("Parents by level count: %llu\n", header->parents_by_level_count);
    
//     // Print parents_by_level
//     printf("Parents by level:\n");
//     for (int i = 0; i < header->parents_by_level_count; i++) {
//         printf("  Level %d: %s\n", i, header->parents_by_level[i]);
//     }
    
//     printf("Hash merkle root: ");
//     for (int i = 0; i < 32; i++) {
//         printf("%02x", (unsigned char)header->hash_merkle_root[i]);
//     }
//     printf("\n");
    
//     printf("Accepted ID merkle root: ");
//     for (int i = 0; i < 32; i++) {
//         printf("%02x", (unsigned char)header->accepted_id_merkle_root[i]);
//     }
//     printf("\n");
    
//     printf("UTXO commitment: ");
//     for (int i = 0; i < 32; i++) {
//         printf("%02x", (unsigned char)header->utxo_commitment[i]);
//     }
//     printf("\n");*/
    
//     printf("Timestamp: %llu\n", header->timestamp);
//     printf("Bits: %u\n", header->bits);
//     printf("Nonce: %llu\n", header->nonce);
//     printf("DAA score: %llu\n", header->daa_score);
//     /*printf("Blue score: %llu\n", header->blue_score);
//     printf("Blue work length: %llu\n", header->blue_work_length);
    
//     printf("Blue work: ");
//     for (int i = 0; i < header->blue_work_length; i++) {
//         printf("%02x", header->blue_work[i]);
//     }
//     printf("\n");
    
//     printf("Pruning point: ");
//     for (int i = 0; i < 32; i++) {
//         printf("%02x", (unsigned char)header->pruning_point[i]);
//     }*/
//     printf("\n");
// }