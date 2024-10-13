struct Uint256 {
    unsigned long long int parts[4];

    __device__ Uint256() {
        parts[0] = parts[1] = parts[2] = parts[3] = 0;
    }

    __device__ Uint256(unsigned long long int value) {
        parts[0] = value;
        parts[1] = parts[2] = parts[3] = 0;
    }

    __device__ Uint256(unsigned long long int p0, unsigned long long int p1, 
                       unsigned long long int p2, unsigned long long int p3) {
        parts[0] = p0;
        parts[1] = p1;
        parts[2] = p2;
        parts[3] = p3;
    }

    __device__ bool operator<=(const Uint256& other) const {
        for (int i = 3; i >= 0; --i) {
            if (parts[i] < other.parts[i]) return true;
            if (parts[i] > other.parts[i]) return false;
        }
        return true;
    }
};

__global__ void init_and_compare_uint256(unsigned long long int* input, Uint256* output, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Uint256 a(input[idx*4], input[idx*4+1], input[idx*4+2], input[idx*4+3]);
    Uint256 b(input[(idx+1)*4], input[(idx+1)*4+1], input[(idx+1)*4+2], input[(idx+1)*4+3]);

    output[idx] = a;
    output[idx+1] = b;
    result[idx] = a <= b;
}