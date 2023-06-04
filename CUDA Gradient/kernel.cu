#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <chrono>

/// =============================================================================================
/// Overall, the code demonstrates how to generate a gradient image using CUDA, 
/// convert it to an OpenCV Mat object, and measure the execution time. It provides 
/// a customizable way to generate gradients with different dimensions and directions on the GPU.
/// =============================================================================================

using namespace std::chrono;

/// <summary>
/// Used to represent the direction of the gradient in the generated image.
/// </summary>
enum GradientDirection
{
    LeftToRight,
    RightToLeft
};

/// <summary>
/// The convertToMat function converts the generated float image data to an OpenCV Mat object
///  of the appropriate data type (CV_32F). It ensures that the image data is copied to a new 
/// memory location to avoid potential memory access issues.
/// </summary>
/// <param name="image">A pointer to float image data.</param>
/// <param name="width">Width of image.</param>
/// <param name="height">Height of image.</param>
/// <returns>The clone() function is used to create a deep copy of the image data and return it as a new Mat object.</returns>
cv::Mat convertToMat(float* image, int width, int height)
{
    cv::Mat matImage(height, width, CV_32F, image);
    return matImage.clone();
}

/// <summary>
/// The generateGradientImageKernel kernel is responsible for calculating the gradient values. 
/// It uses the GPU thread indices to compute the pixel position and assigns a gradient value
/// based on the specified direction (left to right or right to left).
/// </summary>
/// <param name="image">A pointer to float image data.</param>
/// <param name="width">Width of image.</param>
/// <param name="height">Height of image.</param>
/// <param name="direction">The direction of the gradient.</param>
__global__ void generateGradientImageKernel(float* image, int width, int height, GradientDirection direction)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int index = y * width + x;
        float gradientValue = 0.0;

        if (direction == LeftToRight)
            gradientValue = static_cast<float>(x) / width;
        else if (direction == RightToLeft)
            gradientValue = 1.0f - static_cast<float>(x) / width;

        image[index] = gradientValue;
    }
}

/// <summary>
/// The generateGradientImage function generates a gradient image with float values ranging from 0 to 1. 
/// It takes the desired width, height, and gradient direction as parameters. The function allocates
///  memory on the GPU, launches the CUDA kernel, copies the generated image data back to the host, and 
/// returns a pointer to the host memory.
/// </summary>
/// <param name="width">Width of image.</param>
/// <param name="height">Height of image.</param>
/// <param name="direction">The direction of the gradient.</param>
/// <returns>Returns a pointer to the host memory containing the generated image data.</returns>
float* generateGradientImage(int width, int height, GradientDirection direction)
{

    cudaError_t cudaStatus;
    size_t imageSize = width * height * sizeof(float);
    float* deviceImage = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Allocate GPU buffer for deviceImage.
    cudaStatus = cudaMalloc((void**)&deviceImage, imageSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    generateGradientImageKernel <<<numBlocks, threadsPerBlock >>> (deviceImage, width, height, direction);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    float* hostImage = new float[imageSize];
    cudaStatus = cudaMemcpy(hostImage, deviceImage, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(deviceImage);

    return hostImage;
}

int main()
{
    while (true) {

        auto start = high_resolution_clock::now();

        int width = 512;
        int height = 512;

        GradientDirection direction = RightToLeft;  // Change the direction here

        float* gradientImage = generateGradientImage(width, height, direction);

        cv::Mat matImage = convertToMat(gradientImage, width, height);
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        cv::imshow("gradient", matImage);
        cv::waitKey(0);

        delete[] gradientImage;

        
        // To get the value of duration use the count()
        // member function on the duration object
        std::cout << duration.count() << "\xE6s" << '\n';

    }

    return 0;
}
