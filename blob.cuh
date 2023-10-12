//
// Created by didi on 2023/9/15.
//

#ifndef CUDA_PRACTICE_BLOB_CUH
#define CUDA_PRACTICE_BLOB_CUH

# include <array>
# include <string>
# include <iostream>
# include <fstream>
# include <random>
# include <memory>
#include <utility>
#include "utils.cuh"


__global__ void setValueKernel(float* x, float value, int n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = idx * 8;
    if (offset < n) {
        for (int i = 0; i < 8 && offset + i < n; ++i)
            x[offset + i] = value;
    }
}

void setValue(float* x, float value, int n, cudaStream_t stream=nullptr) {
    int blockSize = std::min(512, (n >> 3) + 1);
    int gridSize = n / (blockSize << 3) + 1;
    setValueKernel<<<gridSize, blockSize, 0, stream>>>(x, value, n);
}

typedef enum {CPU, GPU} Device;


class Blob {
private:
    float* gpu_ = nullptr;
    float* cpu_ = nullptr;
    Device device_ = CPU;
    int h_ = 1;
    int w_ = 1;

    void blobFree() {
        if (gpu_ != nullptr) cudaFree(gpu_);
        delete [] cpu_;
    }

    void blobInit(int h, int w) {
        h_ = h;
        w_ = w;
        cpu_ = new float[w * h];
        CUDACHECK(cudaMalloc(&gpu_, sizeof(float) * h * w));
    }

public:

    Blob(int w, float* arr=nullptr) {
        blobInit(1, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }

    Blob(int h, int w, float* arr=nullptr) {
        blobInit(h, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }


    Blob(const std::shared_ptr<Blob>& blob, bool copy=false, int dup=1) {
        blobInit(blob->h(), blob->w());
        if (copy) {
            for (int i = 0; i < dup; ++i)
                cudaMemcpy(gpu() + i * blob->size(),
                           blob->gpu(),
                           blob->memSize(),
                           cudaMemcpyDeviceToDevice);

        } else {
            valueInit();
        }
    }


    void valueInit(float value=0.f) {
        if (device_ == CPU) {
            for (int i = 0; i < size(); ++i)
                cpu()[i] = value;
        } else {
            setValue(gpu(), value, size());
        }
    }

    void arrayInit(const float* arr) {
        memcpy(cpu(), arr, sizeof(float) * size());
    }

    void uniformInit() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float range = std::sqrt(1.f / w());
        std::uniform_real_distribution<> dis(-range, range);

        for (int i = 0; i < size(); ++i)
            cpu()[i] = static_cast<float>(dis(gen));
    }

    void print() {
        printf("\n");
        for (int h = 0; h < h_; ++h) {
            for (int w = 0; w < w_; ++w)
                printf("%6.3f, ", getItem(h, w));
            printf("\n\n");
        }
    }

    void printMem() {
        printf("\n");
        for (int n = 0; n < size(); ++n) {
            printf("%6.4f, ", cpu()[n]);
        }
        printf("\n");
    }

    void printShape() {
        printf("\n");
        printf("(%d, %d)\n", h_, w_);
    }

    ~Blob() {
        blobFree();
    }

    void copy(const std::shared_ptr<Blob>& blob, int dup=1) {
        for (int i = 0; i < dup; ++i) {
            if (blob->device_ == GPU) {
                cudaMemcpy(gpu() + i * blob->size(), blob->gpu(), blob->memSize(), cudaMemcpyDeviceToDevice);
            } else {
                memcpy(cpu() + i * blob->size(), blob->cpu(), blob->memSize());
            }
        }
    }

    float* cpu() {
        if (device_ == GPU) {
            cudaMemcpy(cpu_, gpu_, sizeof(float) * size(), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            device_ = CPU;
        }
        return cpu_;
    }

    float* gpu() {
        if (device_ == CPU) {
            cudaMemcpy(gpu_, cpu_, sizeof(float) * size(), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            device_ = GPU;
        }
        return gpu_;
    }

    float getItem(int h, int w) {
        return cpu()[h * w_ + w];
    }

    float getItem(int w) {
        return cpu()[w];
    }

    void setItem(int h, int w, float v) {
         cpu()[h * w_ + w] = v;
    }

    void setItem(int n, float v) {
        cpu()[n] = v;
    }

    void transpose() {
        if (device_ == CPU) {
            cpu();
            auto cpu = new float[h_ * w_];
            for (int y = 0; y < h_; ++y)
                for (int x = 0; x < w_; ++x)
                    cpu[x * h_ + y] = getItem(y, x);
            cpu_ = cpu;
        } else {

        }
        int tmp = h_;
        h_ = w_;
        w_ = tmp;
    }

    bool equals(const std::shared_ptr<Blob>& b) {
        if (b->h() == this->h() && b->w() == this->w()) {
            for (int y = 0; y < h_; ++y) {
                for (int x = 0; x < w_; ++x) {
                    if (b->getItem(y, x) - this->getItem(y, x) > 0.001)
                        return false;
                }
            }
            return true;
        }
        return false;
    }

    Device device() { return device_; }
    [[nodiscard]] int size() const {return h_ * w_; }
    [[nodiscard]] size_t memSize() const {return size() * sizeof(float); }
    [[nodiscard]] int h() const { return h_; }
    [[nodiscard]] int w() const { return w_; }
};

typedef std::shared_ptr<Blob> blobSp;

#endif //CUDA_PRACTICE_BLOB_CUH
