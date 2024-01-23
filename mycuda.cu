#include <stdio.h>
#include <stdlib.h>


// A simple test of cuda c programming for MPI + OpenMP + CUDA programming
// 一个cuda 函数，不需要输入，调用kernel
__global__ void cadd(float *a, float *b, float *c, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        c[tid] = a[tid] + b[tid];
    }
}
void allocate_memory(float **d_data1, float **d_data2, float **d_result, int size)
{
    cudaMalloc((void **)d_data1, size * sizeof(float));
    cudaMalloc((void **)d_data2, size * sizeof(float));
    cudaMalloc((void **)d_result, size * sizeof(float));
}
void init_data(float *d_data1, float *d_data2, int size)
{
    
    cudaMemset(d_data1, 0, size * sizeof(float));
    cudaMemset(d_data2, 0, size * sizeof(float));

}

extern "C" void cuda_test(int is, float *d_data1, float *d_data2, float *d_result, int size)
{
    // 给d_data1, d_data2, d_result 随机初始化
    // Init data 
    float *h_data1 = (float *)malloc(size * sizeof(float));
    float *h_data2 = (float *)malloc(size * sizeof(float));
    float *h_result = (float *)malloc(size * sizeof(float));

    // for (int i = 0; i < size; i++)
    // {
    //     h_data1[i] = rand() % 100;
    //     h_data2[i] = rand() % 100;
    // }

    // h_data1 从 is * is 到 is * is + size
    // h_data2 从 is + 1 到 is + size + 1

    for (int i = 0; i < size; i++)
    {
        h_data1[i] = is * is + i;
        h_data2[i] = is + i + 1;
    }

    // 将h_data1, h_data2, h_result 拷贝到device
    // Copy h_data1, h_data2, h_result to device
    cudaMemcpy(d_data1, h_data1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data2, size * sizeof(float), cudaMemcpyHostToDevice);

    // 调用kernel
    // Call kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    cadd<<<grid_size, block_size>>>(d_data1, d_data2, d_result, size);

    // 将d_result 拷贝到host
    // Copy d_result to host
    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    // Verify result
    for (int i = 0; i < size; i++)
    {
        if (h_result[i] != h_data1[i] + h_data2[i])
        {
            printf("error: %f + %f != %f\n", h_data1[i], h_data2[i], h_result[i]);
            break;
        }
    }
    // 打印第一个cadd的参数和结果
    // Print the first cadd's parameters and result
    // printf("cadd: %f + %f = %f\n", h_data1[0], h_data2[0], h_result[0]);

    // 写入test_is.txt, is为输入参数
    // Write test_is.txt, is is the input parameter
    char filename[20];
    sprintf(filename, "test/test_%d.txt", is);
    FILE *fp = fopen(filename, "w");
    // 写入所有测试数据
    // Write all test data
    for (int i = 0; i < size; i++)
    {
        fprintf(fp, "%f %f %f\n", h_data1[i], h_data2[i], h_result[i]);
    }
    fclose(fp);




    free(h_data1);
    free(h_data2);
    free(h_result);


}