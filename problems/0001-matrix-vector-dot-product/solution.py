#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    // 获取当前线程的全局 ID
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

std::vector<float> matrix_dot_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    if (matrix.empty() || matrix[0].empty() || matrix[0].size() != vec.size()) {
        return {-1};
    }

    int rows = matrix.size();
    int cols = vec.size();
    
    std::vector<float> flat_matrix(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat_matrix[i * cols + j] = matrix[i][j];
        }
    }

    // 1. 分配统一内存 (Unified Memory)
    float* d_matrix;
    float* d_vector;
    float* d_result;
    cudaMallocManaged(&d_matrix, rows * cols * sizeof(float));
    cudaMallocManaged(&d_vector, cols * sizeof(float));
    cudaMallocManaged(&d_result, rows * sizeof(float));

    // 2. 直接赋值（统一内存 CPU 可直接访问，不需要 cudaMemcpy）
    for (int i = 0; i < rows * cols; ++i) {
        d_matrix[i] = flat_matrix[i];
    }
    for (int i = 0; i < cols; ++i) {
        d_vector[i] = vec[i];
    }

    // 3. 启动核函数
    int threadsPerBlock = 256;
    // 向上取整，保证分配的线程数 >= rows
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result, rows, cols);

    // 4. 等待 GPU 计算完成
    cudaDeviceSynchronize();

    // 5. 直接在 CPU 端读取结果
    std::vector<float> result(rows); 
    for (int i = 0; i < rows; ++i) {
        result[i] = d_result[i];
    }

    // 6. 释放统一内存
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return result;
}