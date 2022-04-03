#include "Kernels.cuh"
#include <math.h>
#include <iostream>

typedef struct TestStruct
{
    float x;
    float y;
    float z;
} TestStruct;

__global__
void add(int n, TestStruct* x, TestStruct* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i].x = x[i].x + y[i].x;
        y[i].y = x[i].y + y[i].y;
        y[i].z = x[i].z + y[i].z;
    }
}

__global__
void collisionCheck(int n, glm::vec3* oldPositions, glm::vec3* currentPositions, glm::vec3* finalPositions)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        glm::vec3 pos = currentPositions[index];
        float totalRadius = 1.0f;

        for (int i = 0; i < n; i++) {
            if (i != index) {
                glm::vec3 other = currentPositions[i];

                float xDiff = pos.x - other.x;
                float yDiff = pos.y - other.y;
                float zDiff = pos.z - other.z;

                float distance = sqrtf(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

                if (distance < totalRadius) {
                    finalPositions[index] = oldPositions[index];
                    return;
                }
            }
        }

        finalPositions[index] = pos;
    }
}

#define GRAVITY 0.05f

__global__
void applyForces(int n, glm::vec3* positions)
{

}

void Kernels::TestKernel()
{
    int N = 1 << 20;
    TestStruct* x, * y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(TestStruct));
    cudaMallocManaged(&y, N * sizeof(TestStruct));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i].x = 1.0f;
        x[i].y = 1.0f;
        x[i].z = 1.0f;
        y[i].x = 2.0f;
        y[i].y = 2.0f;
        y[i].z = 2.0f;
    }
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N * sizeof(TestStruct), device, NULL);
    cudaMemPrefetchAsync(y, N * sizeof(TestStruct), device, NULL);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // Run kernel on 1M elements on the GPU
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();
    

    // Wait for GPU to finish before accessing on host
    

    // Check for errors (all values should be 3.0f)
    float maxError1 = 0.0f;
    float maxError2 = 0.0f;
    float maxError3 = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError1 = fmax(maxError1, fabs(y[i].x - 3.0f));
        maxError2 = fmax(maxError1, fabs(y[i].y - 3.0f));
        maxError3 = fmax(maxError1, fabs(y[i].z - 3.0f));
    }
    std::cout << "Max error: " << maxError1 << " " << maxError2 << " " << maxError3 << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}

void Kernels::CollisionCheck(glm::vec3** oldPositions, glm::vec3** currentPositions, glm::vec3** finalPositions, int arraySize) {
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(oldPositions, arraySize * sizeof(glm::vec3), device, NULL);
    cudaMemPrefetchAsync(currentPositions, arraySize * sizeof(glm::vec3), device, NULL);
    cudaMemPrefetchAsync(finalPositions, arraySize * sizeof(glm::vec3), device, NULL);

    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    collisionCheck<<<numBlocks, blockSize>>>(arraySize, *oldPositions, *currentPositions, *finalPositions);
    cudaDeviceSynchronize();
}

void Kernels::CollisionCheckCPU(glm::vec3* oldPositions, glm::vec3* currentPositions, glm::vec3* finalPositions, int arraySize) {
    float totalRadius = 1.0f;

    for (int i = 0; i < arraySize; i++) {
        glm::vec3 pos = currentPositions[i];
        bool changeFinalPos = true;

        for (int j = 0; j < arraySize; j++) {
            if (i != j) {
                glm::vec3 other = currentPositions[j];

                float xDiff = pos.x - other.x;
                float yDiff = pos.y - other.y;
                float zDiff = pos.z - other.z;

                float distance = sqrtf(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

                if (distance < totalRadius) {
                    finalPositions[i] = oldPositions[i];
                    changeFinalPos = false;
                    return;
                }
            }
        }

        if (changeFinalPos) {
            finalPositions[i] = pos;
        }
    }
}