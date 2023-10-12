//
// Created by didi on 2023/9/15.
//

#ifndef CUDA_PRACTICE_KERNEL_CUH
#define CUDA_PRACTICE_KERNEL_CUH

#include "blob.cuh"
#define OFS(y, x, n) ((y) * (n) + (x))

// blocksize == 16 * 16 == smem size
// Every thread process 4 numbers in x and y directs respectively
__global__ void transposeKernel(const float* a, float* b, int m, int n) {
    uint stride = 4;
    uint ty = threadIdx.y;
    uint tx = threadIdx.x;
    uint oy = blockDim.y * stride * blockIdx.y;
    uint ox = blockDim.x * stride * blockIdx.x;
    static __shared__ float smem[2][16][16 + 1];

    for (uint y = 0; y < stride; ++y) {
        for (uint x = 0; x < stride; ++x) {
            if (oy + y * blockDim.y + ty < m && ox + x * blockDim.x + tx < n)
                smem[x & 1][tx][ty] = a[OFS(oy + y * blockDim.y + ty, ox + x * blockDim.x + tx, n)];
            __syncthreads();
            if (ox + x * blockDim.x + ty < n && oy + y * blockDim.y + tx < m)
                b[OFS(ox + x * blockDim.x + ty, oy + y * blockDim.y + tx, m)] = smem[x & 1][ty][tx];

        }
    }
}

void transpose(const blobSp& A, const blobSp& B) {
    int m = A->h(), n = A->w();
    dim3 blockSize;
    blockSize.y = 16;
    blockSize.x = 16;
    dim3 gridSize;
    gridSize.y = (m / blockSize.y / 4) + 1;
    gridSize.x = (n / blockSize.x / 4) + 1;

    transposeKernel<<<gridSize, blockSize>>>(A->gpu(), B->gpu(), m, n);
    cudaDeviceSynchronize();
}

// 每个线程处理4个float数据，数据之间相距blockSize
#define IPT 4
__global__ void reduceSumKernel(float* a, int n) {
    float buf = 0.f;
    uint offset = blockDim.x * blockIdx.x * IPT;
    extern __shared__ float smem[];

    for (int step = 0; step < IPT; ++step) {
        if (offset + threadIdx.x < n) {
            buf += a[offset + threadIdx.x];
            offset += blockDim.x;
        }
    }
    smem[threadIdx.x] = buf;
    __syncthreads();

    for (uint stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        a[blockIdx.x] = smem[0];
}

void reduceSum(float* x, float* res, int n) {
    float* copy;
    CUDACHECK(cudaMalloc(&copy, sizeof(float) * n));
    CUDACHECK(cudaMemcpy(copy, x, sizeof(float) * n, cudaMemcpyDeviceToDevice));

    int blockSize = 256;
    int gridSize;
    while (n > 1) {
        gridSize = n / (blockSize * IPT) + 1;
        reduceSumKernel<<<gridSize, blockSize, blockSize>>>(copy, n);
        cudaDeviceSynchronize();
        n = gridSize;
    }
    cudaMemcpy(res, copy, sizeof(float), cudaMemcpyDeviceToHost);
}

__constant__ float constantArray[128];

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
template <
        const int BM,
        const int BN,
        const int BK,
        const int RM,
        const int RN
>
__global__ void gemmKernel(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float as[2][BK][BM];
    __shared__ float bs[2][BK][BN];

    unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int offsetA = blockIdx.y * BM;
    unsigned int offsetB = blockIdx.x * BN;
    unsigned int offsetTile;

    float a[2][RM];
    float b[2][RN];
    float c[RM][RN] = {0.f};

    uint8_t nThreadPerRowA = BK / RM, nThreadPerRowB = BN / RN;
    uint8_t ay = tid / nThreadPerRowA;
    uint8_t ax = tid & (nThreadPerRowA - 1);
    uint8_t by = tid / nThreadPerRowB;
    uint8_t bx = tid & (nThreadPerRowB - 1);
    int r;

    for (int kIter = 0; kIter < K / BK; ++kIter) {
        r = kIter & 1;
        offsetTile = kIter * BK;

        FLOAT4(a[0][0]) = FLOAT4(A[OFS(offsetA + ay, offsetTile + ax * RM, K)]);
        FLOAT4(a[0][4]) = FLOAT4(A[OFS(offsetA + ay, offsetTile + ax * RM + 4, K)]);
        for (int i = 0; i < RM; ++i)
            as[r][ax * RM + i][ay] = a[0][i];

        FLOAT4(b[0][0]) = FLOAT4(B[OFS(offsetTile + by, offsetB + bx * RN, N)]);
        FLOAT4(b[0][4]) = FLOAT4(B[OFS(offsetTile + by, offsetB + bx * RN + 4, N)]);
        for (int i = 0; i < RN; i += 4)
            FLOAT4(bs[r][by][bx * RN + i]) = FLOAT4(b[0][i]);
        __syncthreads();

        // compute
        for (int bk = 0; bk < BK; ++bk) {
            for (int j = 0; j < RM; ++j)
                a[r][j] = as[r][bk][RM * threadIdx.y + j];
            for (int j = 0; j < RN; ++j)
                b[r][j] = bs[r][bk][RN * threadIdx.x + j];
            for (int i = 0; i < RM; ++i)
                for (int j = 0; j < RN; ++j)
                    c[i][j] += a[r][i] * b[r][j];
        }
    }
    for (int i = 0; i < RM; ++i)
        for (int j = 0; j < RN; j += 4)
            FLOAT4(C[OFS(ty * RM + i, tx * RN + j, N)]) = FLOAT4(c[i][j]);
}

void gemm(float* A, float* B, float* C, int M, int K, int N) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int RM = 8;
    const int RN = 8;
    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = 8;
    blockSize.y = 8;
    gridSize.y = M /  blockSize.y / RM;
    gridSize.x = N /  blockSize.x / RN;
    gemmKernel<BM, BN, BK, RM, RN><<<gridSize, blockSize>>>(A, B, C, M, K, N);
}

//template<
//        const int N,
//        const int IC, const int OC,
//        const int IH, const int IW,
//        const int KH, const int KW,
//        const int OH, const int OW,
//        const int GM, const int GK, const int GN,
//        const int TM, const int TK, const int TN
//>
//__global__ void im2colKernel(float* x, float* w, float* y) {
//
//    int n, ic, oc, ih, iw, kh, kw, oh, ow, jr, kr;
//    float acc[TM][TN] = {0.f};
//    __shared__ float smem_x[TM][TK];
//    __shared__ float smem_w[TK][TN];
//
//    for (int out_i = 0; out_i < GM; out_i += TM) {
//        for (int out_j = 0; out_j < GN; out_j += TN) {
//            for (int in_i = out_i; in_i < TM; ++in_i) {
//                oc = in_i;
//
//                for (int in_j = out_j; in_j < TN; ++in_j) {
//                    n = in_j / (OH * OW);
//                    jr = in_j % (OH * OW);
//                    oh = jr / OW;
//                    ow = jr % OW;
//
//                    for (int out_k = 0; out_k < GK; out_k += TK) {
//                        // load to smem
//                        for (int in_k = out_k; in_k < TK; ++in_k) {
//                            ic = in_k / (KH * KW);
//                            kr = in_k % (KH * KW);
//                            kh = kr / KW;
//                            kw = kr % KW;
//                            iw = ow + kw - 1;
//                            ih = oh + kh - 1;
//
//                            acc[in_i][in_j] += smem_x[in_i][in_k] * smem_w[in_k][in_j];
//                        }
//                    }
//
//                    y(n, oc, oh, ow) = acc[in_i][in_j];
//                }
//            }
//        }
//    }
//}

// blockDim.x = 128 = 16 * 8
// 每个线程处理8个数据，偏移为2^3 = 8
template <
        const int BM,   // 128
        const int BN,   // 128
        const int BK,   // 8
        const int RM,   // 8
        const int RN,   // 8
        const int S     // 线程处理步长，2
>
__global__ void sgemm(float* a, float* b, float* c, int m, int n, int k) {
    const uint NPT = 4;

    __shared__ float acc[BM][BN];
    __shared__ float smem_a[BK][BM + 1];
    __shared__ float smem_b[BK][BN + 1];

    float reg_a[RM];
    float reg_b[RN];

    uint n_threads = threadIdx.y * blockDim.x + threadIdx.x;
    uint out_i = blockIdx.y * blockDim.y;   // gridDim.y = m / BM
    uint out_j = blockIdx.x * blockDim.x;   // gridDim.x = n / BN
    uint t_ofs = n_threads * BK;
    uint s_row_a = t_ofs / BK;
    uint s_col_a = t_ofs % BK;
    uint s_row_b = t_ofs / BN;
    uint s_col_b = t_ofs % BN;
    uint s_ofs_a = threadIdx.y * 8;
    uint s_ofs_b = threadIdx.x * 8;


    for (int out_k = 0; out_k < k; out_k += BK) {
        // load to smem
        for (uint s = 0; s < S; ++s) {
            for (uint i = 0; i < NPT; ++i)
                smem_a[s_col_a + s * NPT + i][s_row_a] = a[OFS(out_i, out_k + s * NPT + i, k)];
            FLOAT4(smem_b[s_row_b][s_col_b + s * IPT]) = FLOAT4(b[OFS(out_k + s_row_b, out_j + s_col_b + s * IPT, n)]);
        }
        __syncthreads();

        // load to reg
        for (int in_k = 0; in_k < BK; ++in_k) {
            FLOAT4(reg_a[0]) = FLOAT4(smem_a[in_k][s_ofs_a]);
            FLOAT4(reg_a[4]) = FLOAT4(smem_a[in_k][s_ofs_a + 4]);
            FLOAT4(reg_b[0]) = FLOAT4(smem_b[in_k][s_ofs_b]);
            FLOAT4(reg_b[4]) = FLOAT4(smem_b[in_k][s_ofs_b + 4]);
        }

        for (int in_i = 0; in_i < BM; ++in_i) {
            for (int in_j = 0; in_j < BN; ++in_j) {
                for (int in_k = 0; in_k < BK; ++in_k) {
                    acc[in_i][in_j] += smem_a[in_k][in_i] * smem_b[in_k][in_j];
                }
            }
        }
    }

    for (int in_i = 0; in_i < BM; ++in_i) {
        for (int in_j = 0; in_j < BN; ++in_j) {
            c[OFS(out_i + in_i, out_j + in_j, n)] = acc[in_i][in_j];
        }
    }

}


__global__ void reduce_max_kernel(float* x, int n) {
    uint ofs = gridDim.x * blockDim.x;
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    float buffer[4];
    extern __shared__ float smem[];

    for (uint i = 0; i < 4; ++i) {
        buffer[i] = -1e10;
        if (i * ofs + idx < n)
            buffer[i] = x[i * ofs + idx];
    }

    for (uint i = 1; i < 4; ++i)
        buffer[0] = max(buffer[0], buffer[i]);
    smem[threadIdx.x] = buffer[0];
    __syncthreads();

    for (uint stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        x[blockIdx.x] = smem[0];
}

std::shared_ptr<Blob> reduce_max(std::shared_ptr<Blob>& x) {
    auto copy = std::make_shared<Blob>(x, true);
    auto res = std::make_shared<Blob>(1);
    int blockSize = 512;
    int gridSize;
    int n = x->size();

    while (n > 1) {
        if (n <= blockSize << 2) {
            blockSize = n;
            gridSize = 1;
        } else {
            gridSize = (n / (blockSize << 2)) + 1;
        }
        reduce_max_kernel<<<gridSize, blockSize, blockSize>>>(copy->gpu(), n);
        n = gridSize;
    }

    cudaMemcpy(res->gpu(), copy->gpu(), sizeof(float), cudaMemcpyDeviceToDevice);

    return res;
}

#define FP4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__global__ void vector_exp_kernel(float* x, float a, float b, int n) {
    float buffer[8];
    uint idx = (blockDim.x * blockIdx.x + threadIdx.x) << 3;

    for (uint s = 0; s < 8; s += 4) {
        if (idx < n) {
            FLOAT4(buffer[s]) = FLOAT4(x[idx]);
            for (uint i = 0; i < 4; ++i)
                buffer[s + i] = exp(buffer[s + i] + a * b);
            FLOAT4(x[idx]) = FLOAT4(buffer[s])
        }
    }
}

void vector_exp(const std::shared_ptr<Blob>& x, float a, float b) {
    int blockSize = 512;
    int gridSize = x->size() / (blockSize << 3) + 1;
    vector_exp_kernel<<<blockSize, gridSize>>>(x->gpu(), a, b, x->size());
}


void softmax(std::shared_ptr<Blob> x) {
    auto x_max = reduce_max(x);
    vector_exp(x, x_max->getItem(0), -1);
}

#endif //CUDA_PRACTICE_KERNEL_CUH
