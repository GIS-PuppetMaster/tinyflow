#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <math.h>
//��
#include <cudnn.h>
#include <stdlib.h>
#include <iostream>

//��-
using namespace std;
#define MAX_THREADS_NUM 1024
#define MAX_BLOCKS_NUM 4096
#define BLOCK_NUM(count) min(((count + MAX_THREADS_NUM - 1) / MAX_THREADS_NUM), MAX_BLOCKS_NUM)
#define BLOCK_NUM_2(count,threads) min(((count + threads - 1) / threads), MAX_BLOCKS_NUM)
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)
#define CUDA_1D_KERNEL_LOOP_2(i, n) \
  for (size_t i = blockIdx.x ; i < (n); \
        i += gridDim.x)
#define CUDA_1D_KERNEL_LOOP_3(i, n) \
  for (size_t i = threadIdx.x ; i < (n); \
        i += blockDim.x)

//��
#define CUDNN_CALL(f) { \
cudnnStatus_t err = (f); \
if (err != CUDNN_STATUS_SUCCESS) {\
    \
        std::cout << "    Error occurred: " << err << std::endl; \
        std::exit(1); \
} \
}

//��-
__global__ void matrix_array_set_kernel(int count,
                                        float *arr,
                                        float value) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    arr[index] = value;
  }
}

__global__ void matrix_broadcast_to_kernel0(int inputCount, float* inputArr,
                                           int outputCount, float* outputArr) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      outputArr[index] = inputArr[index % inputCount];
  }
}

__global__ void matrix_broadcast_to_backward_kernel0(float* inputArr,
                                                    int outputCount, float* outputArr,
                                                    int moreDim) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {
        float sum = 0;
        for(int i=0; i < moreDim; i++) {
            sum += inputArr[index + i * outputCount];
        }
        outputArr[index] = sum;
    }
}


__global__ void matrix_broadcast_to_backward_kernel0_o(float* inputArr,
                                                        int outputCount, float* outputArr,
                                                        int moreDim) {

    __shared__ float  tmp[MAX_THREADS_NUM];
    CUDA_1D_KERNEL_LOOP_2(i, outputCount) {

        int sum = 0;

        CUDA_1D_KERNEL_LOOP_2(j, moreDim) {

            tmp[threadIdx.x] = inputArr[i+j*outputCount];

            __syncthreads();


            for (int index = 1; index < blockDim.x; index *= 2)        //归约求和
            {
                if (threadIdx.x % (2 * index) == 0 && j+index < moreDim)
                {
                    tmp[threadIdx.x] += tmp[threadIdx.x+index];
                }
                __syncthreads();
            }
            
            if(threadIdx.x == 0){

                sum += tmp[0];
            }

        }

        if(threadIdx.x == 0){
            outputArr[i] = sum;
        }


    }
}


__global__ void matrix_broadcast_to_kernel11(float* inputArr,
                                            int outputCount, float* outputArr,
                                            int outputlow) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {
        outputArr[index] = inputArr[index / outputlow];
    }
}

__global__ void matrix_broadcast_to_kernel12(int inputCount, float* inputArr,
                                            int outputCount, float* outputArr) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {
        if(index >= inputCount){
            outputArr[index] = inputArr[index % inputCount];
        }
    }
}

__global__ void matrix_broadcast_to_backward_kernel1(float* inputArr,
                                                    int outputCount, float* outputArr,
                                                    int moreDim) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {
        float sum = 0;
        for(int i=0; i < moreDim; i++) {
            sum += inputArr[index*moreDim + i];
        }
        outputArr[index] = sum;
    }
}


__global__ void matrix_broadcast_to_backward_kernel1_o(float* inputArr,
                                                        int outputCount, float* outputArr,
                                                        int moreDim) {

    __shared__ float  tmp[MAX_THREADS_NUM];
    CUDA_1D_KERNEL_LOOP_2(i, outputCount) {

        int sum = 0;

        CUDA_1D_KERNEL_LOOP_2(j, moreDim) {

            tmp[threadIdx.x] = inputArr[i*moreDim+j];

            __syncthreads();


            for (int index = 1; index < blockDim.x; index *= 2)        //归约求和
            {
                if (threadIdx.x % (2 * index) == 0 && j+index < moreDim)
                {
                    tmp[threadIdx.x] += tmp[threadIdx.x+index];
                }
                __syncthreads();
            }
            
            if(threadIdx.x == 0){

                sum += tmp[0];
            }

        }

        if(threadIdx.x == 0){
            outputArr[i] = sum;
        }


    }
}


__global__ void matrix_reduce_sum_axis_zero_kernel(float* inputArr,
                                                   int outputCount, float* outputArr,
                                                   int zeroDim) {
      CUDA_1D_KERNEL_LOOP(index, outputCount) {
          float sum = 0;
          for (int i = 0; i < zeroDim; ++i) {
              sum += inputArr[index + i * outputCount];
          }
          outputArr[index] = sum;
      }
}



__global__ void matrix_elementwise_add_kernel(float* matAData, float* matBData,
                                              float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] + matBData[index];
    }
}

__global__ void matrix_elementwise_add_by_const_kernel(float* inputArr, float val,
                                                       float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] + val;
    }
}

__global__ void matrix_elementwise_multiply_kernel(float* matAData, float* matBData,
                                                   float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] * matBData[index];
    }
}

__global__ void matrix_elementwise_multipy_by_const_kernel(float* inputArr, float val,
                                                           float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] * val;
    }
}

__global__ void matrix_relu_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index];
        if (inputArr[index] < 0) {
            outputArr[index] = 0.f;
        }
    }
}

__global__ void matrix_relu_gradient_kernel(const float* inputArr, const float* gradArr,
                                            float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] > 0 ? gradArr[index] : 0;
    }
}

__global__ void matrix_softmax_kernel(int nRow, int nCol, float* inputArr, float* outputArr) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= nRow) return;

    float* input = inputArr + y * nCol;
    float* output = outputArr + y * nCol;

    float maxval = *input;
    for (int i = 1; i < nCol; ++i) {
        maxval = max(input[i], maxval);
    }
    float sum = 0;
    for (int i = 0; i < nCol; ++i) {
        sum += expf(input[i] - maxval);
    }
    for (int i = 0; i < nCol; ++i) {
        output[i] = expf(input[i] - maxval) / sum;
    }
}

/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void matrix_exp_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = exp(inputArr[index]);
    }
}

__global__ void matrix_log_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = log(inputArr[index]);
    }
}

__global__ void matrix_reverse_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = 1. / inputArr[index];
    }
}

__global__ void matrix_pow_kernel(float* inputArr, float val, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = pow(inputArr[index],val);
    }
}




int DLGpuArraySet(DLArrayHandle arr, float value, void **cudaStream) {
  int count = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    count *= arr->shape[i];
  }
  float *arr_data = (float *)arr->data;
  matrix_array_set_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    count, arr_data, value);
  cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuBroadcastTo0(const DLArrayHandle input, DLArrayHandle output, void **cudaStream) {
    //assert(input->ndim + 1 == output->ndim);
    int inputCount = 1, outputCount = 1;
    for (int i = 0; i < input->ndim; ++i) {
        inputCount *= input->shape[i];
        if(output->ndim==1){
            assert(input->ndim==1 && input->shape[0]==1);
        }else if(input->ndim==1 && input->shape[0]==1){
            continue;
        }
        else{
            assert(input->shape[(input->ndim)-1-i] == output->shape[(output->ndim)-1-i]);
        }
    }
    for (int i = 0; i < output->ndim; ++i) {
        outputCount *= output->shape[i];
    }
 
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_broadcast_to_kernel0<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    inputCount, inputArr, outputCount, outputArr);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuBroadcastToBackward0(const DLArrayHandle input, DLArrayHandle output) {
    //assert(input->ndim + 1 == output->ndim);
    int outputCount = 1;
    int inputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {
        outputCount *= output->shape[i];
        inputCount *= input->shape[i];
        if(input->ndim==1){
            assert(output->ndim==1 && output->shape[0]==1);
        }
        else if(output->ndim==1 && output->shape[0]==1){
            continue;
        }
        else{
            assert(input->shape[(input->ndim)-1-i] == output->shape[(output->ndim)-1-i]);
        }
    }
    for (int i = output->ndim; i < input->ndim; ++i) {
        inputCount *= input->shape[i];
    }
 
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;

    int tmpoutputCount = inputCount/input->shape[0];


    






    if((input->ndim) == 1 ||(input->ndim) == 2){
        // matrix_broadcast_to_backward_kernel0<<<BLOCK_NUM(tmpoutputCount), MAX_THREADS_NUM>>>(
        //     inputArr, tmpoutputCount, outputArr,input->shape[0]);
        matrix_broadcast_to_backward_kernel0_o<<<tmpoutputCount, MAX_THREADS_NUM>>>(
            inputArr, tmpoutputCount, outputArr,input->shape[0]);
    }
    else{
        float* tmpArr;
        cudaMalloc((void**)&tmpArr, sizeof(float)*tmpoutputCount);

        // matrix_broadcast_to_backward_kernel0<<<BLOCK_NUM(tmpoutputCount), MAX_THREADS_NUM>>>(
        //     inputArr, tmpoutputCount, tmpArr,input->shape[0]);

        // printf("%d\n",tmpoutputCount);
        matrix_broadcast_to_backward_kernel0_o<<<tmpoutputCount, MAX_THREADS_NUM>>>(
            inputArr, tmpoutputCount, tmpArr,input->shape[0]);
    
        int i;
        for(i=2;i < input->ndim;i++){
    
            tmpoutputCount /= input->shape[i];
            if(tmpoutputCount <= outputCount){
                break;
            }
    
            // matrix_broadcast_to_backward_kernel0<<<BLOCK_NUM(tmpoutputCount), MAX_THREADS_NUM>>>(
            //     tmpArr, tmpoutputCount, tmpArr,input->shape[i]);

            matrix_broadcast_to_backward_kernel0_o<<<tmpoutputCount, MAX_THREADS_NUM>>>(
                tmpArr, tmpoutputCount, tmpArr,input->shape[i]);
    
        }    
        // matrix_broadcast_to_backward_kernel0<<<BLOCK_NUM(tmpoutputCount), MAX_THREADS_NUM>>>(
        //     tmpArr, tmpoutputCount, outputArr,input->shape[i]);
        // printf("%d\n",tmpoutputCount);
        matrix_broadcast_to_backward_kernel0_o<<<tmpoutputCount, MAX_THREADS_NUM>>>(
            tmpArr, tmpoutputCount, outputArr,input->shape[i]);

        cudaFree(tmpArr);

    }
    
    cudaDeviceSynchronize();
    return 0;
}


int DLGpuBroadcastTo1(const DLArrayHandle input, DLArrayHandle output, void **cudaStream) {

    assert(input->ndim  == 1 &&input->shape[0] == output->shape[1]);
    int outputlow = 1;
    
    for (int i = 2; i < output->ndim; ++i) {
        outputlow *= output->shape[i];
    }
    int count1 = outputlow*output->shape[1];
    int count2 = count1 * output->shape[0];

    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_broadcast_to_kernel11<<<BLOCK_NUM(count1), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    inputArr, count1, outputArr, outputlow);

    matrix_broadcast_to_kernel12<<<BLOCK_NUM(count2), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    count1, outputArr, count2, outputArr);

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuBroadcastToBackward1(const DLArrayHandle input, DLArrayHandle output) {
    //assert(input->ndim + 1 == output->ndim);
    assert(output->ndim  == 1 &&output->shape[0] == input->shape[1]);

    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;

    int outcn = 1;
    for (int i = 1; i < input->ndim; ++i) {
        outcn *= input->shape[i];
    }

    float* tmpArr,*tmpArr2;
    cudaMalloc((void**)&tmpArr, sizeof(float)*outcn);
    //n

    // matrix_broadcast_to_backward_kernel0<<<BLOCK_NUM(outcn), MAX_THREADS_NUM>>>(
    //     inputArr, outcn, tmpArr,input->shape[0]);

    matrix_broadcast_to_backward_kernel0_o<<<outcn, MAX_THREADS_NUM>>>(
        inputArr, outcn, tmpArr,input->shape[0]);

    //hw
    outcn = outcn/input->shape[(input->ndim)-1];
    cudaMalloc((void**)&tmpArr2, sizeof(float)*outcn);

    for(int i=(input->ndim)-1;i>2;i--){

        // matrix_broadcast_to_backward_kernel1<<<BLOCK_NUM(outcn), MAX_THREADS_NUM>>>(
        //     tmpArr, outcn, tmpArr2,input->shape[i]);

        matrix_broadcast_to_backward_kernel1_o<<<outcn, MAX_THREADS_NUM>>>(
            tmpArr, outcn, tmpArr2,input->shape[i]);

        outcn = outcn/input->shape[i-1];
        float *tmptmp = tmpArr2;
        tmpArr2 = tmpArr;
        tmpArr = tmptmp;
    }

    // matrix_broadcast_to_backward_kernel1<<<BLOCK_NUM(outcn), MAX_THREADS_NUM>>>(
    //     tmpArr, outcn, outputArr,input->shape[2]);

    matrix_broadcast_to_backward_kernel1_o<<<outcn, MAX_THREADS_NUM>>>(
        tmpArr, outcn, outputArr,input->shape[2]);

    cudaFree(tmpArr);
    cudaFree(tmpArr2);

    cudaDeviceSynchronize();
    return 0;
}


__global__ void matrix_reduce_sum_axis_n_kernel(float* inputArr,
                                                int outputCount, float* outputArr,
                                                int reduceDim,int stride,int lowstride) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {
        int lown = index / stride;
        int lown1 = index % stride;
        float sum = 0;
        for (int i = 0; i < reduceDim; ++i) {
        sum += inputArr[lown * lowstride + lown1 + i * stride];
        }
        outputArr[index] = sum;
    }
}

__global__ void matrix_reduce_sum_axis_n_kernel_backward(float* inputArr,
                                                        int outputCount, float* outputArr,
                                                        int reduceDim,int lowstride) {
    CUDA_1D_KERNEL_LOOP(index, outputCount) {

        int lown = index / lowstride / reduceDim;
        int lown1 = index % lowstride;
        outputArr[index] += inputArr[lown * lowstride + lown1];

    }
}



int DLGpuReduceSumAxisN(const DLArrayHandle input, DLArrayHandle output, const int axis) {


    if(input->ndim == 1){
    assert(1 == output->ndim);
    }else{
    assert(input->ndim == output->ndim + 1);
    }

    int stride = 1;

    for (int i = input->ndim; i > axis + 1; --i) {

        stride = stride * (input->shape[i-1]);
    }
    int reduceDim = input->shape[axis], outputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {

        if( i < axis){
            assert(input->shape[i] == output->shape[i]);
        }else if(input->ndim != 1){
            assert(input->shape[i+1] == output->shape[i]);
        }

        outputCount *= output->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
   // printf("%d",reduceDim);
   // printf("%d",outputCount);
   // printf("%d",stride);
    int lowstride = reduceDim * stride;
    matrix_reduce_sum_axis_n_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
            inputArr, outputCount, outputArr, reduceDim, stride, lowstride);

    cudaDeviceSynchronize();
    return 0;
}

int DLGpuReduceSumAxisNBackward(const DLArrayHandle input, DLArrayHandle output, const int axis) {


    if(output->ndim == 1){
        assert(1 == input->ndim);
    }else{
        assert(input->ndim +1== output->ndim);
    }

    int lowstride = 1;

    for (int i = (output->ndim) - 1; i > axis; --i) {
        lowstride = lowstride * (output->shape[i]);
    }

    int reduceDim = output->shape[axis], outputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {



        outputCount *= output->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_reduce_sum_axis_n_kernel_backward<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
        inputArr, outputCount, outputArr, reduceDim, lowstride);

    cudaDeviceSynchronize();
    return 0;
}



//new
int DLGReduceSumGetCudnnlist(const int *input_shapes,
    const int *output_shapes,
    const int sizeofshape,
    cudnnTensorFormat_t dataformat,
    void *** cudnnlist,
    void **cudnnHandle){


    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_n;
    int output_c;
    int output_h;
    int output_w;

    if(sizeofshape == 3){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = 1;
            input_w = input_shapes[2];

            output_n = output_shapes[0];
            output_c = output_shapes[1];
            output_h = 1;
            output_w = output_shapes[2];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[2];
            input_h = 1;
            input_w = input_shapes[1];

            output_n = output_shapes[0];
            output_c = output_shapes[2];
            output_h = 1;
            output_w = output_shapes[1];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            output_n,
            output_c,
            output_h,
            output_w));
    }
    if(sizeofshape == 4){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = input_shapes[2];
            input_w = input_shapes[3];

            output_n = output_shapes[0];
            output_c = output_shapes[1];
            output_h = output_shapes[2];
            output_w = output_shapes[3];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[3];
            input_h = input_shapes[1];
            input_w = input_shapes[2];

            output_n = output_shapes[0];
            output_c = output_shapes[3];
            output_h = output_shapes[1];
            output_w = output_shapes[2];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            output_n,
            output_c,
            output_h,
            output_w));
    }
    if(sizeofshape == 5){

        int* input_shape;
        input_shape = (int*)malloc(sizeof(int) * 5);
        for(int i=0;i<5;i++){
            input_shape[i]= input_shapes[i];
        }
        int* output_shape;
        output_shape = (int*)malloc(sizeof(int) * 5);
        for(int i=0;i<5;i++){
            output_shape[i]= output_shapes[i];
        }

        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            5,
            input_shape));
        //output
        CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            5,
            output_shape));
    }

    //reduce
    cudnnReduceTensorDescriptor_t reduceTensorDesc;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduceTensorDesc));

    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        reduceTensorDesc,
        CUDNN_REDUCE_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));


    size_t *IndicesSize = (size_t *)malloc(sizeof(size_t));
    size_t *WorkspaceSize = (size_t *)malloc(sizeof(size_t));

    CUDNN_CALL(cudnnGetReductionIndicesSize(
        handle,
        reduceTensorDesc,
        input_descriptor,
        output_descriptor,
        IndicesSize));

    CUDNN_CALL(cudnnGetReductionWorkspaceSize(
        handle,
        reduceTensorDesc,
        input_descriptor,
        output_descriptor,
        WorkspaceSize));



    * cudnnlist = (void **)malloc(sizeof(void *)*5);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = output_descriptor;
    (*cudnnlist)[2] = reduceTensorDesc;
    (*cudnnlist)[3] = IndicesSize;
    (*cudnnlist)[4] = WorkspaceSize;

    //CUDNN_CALL(cudnnDestroy(handle));

    return 0;




}



//new
int DLGpuReduceSum(const DLArrayHandle input, DLArrayHandle output, void ***cudnnlist, void ** cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace){

    void* indices= nullptr;
    void* workspace= nullptr;

    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);
    
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[1]);

    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)((*cudnnlist)[2]);

    size_t *IndicesSize = (size_t *)((*cudnnlist)[3]);

    size_t *WorkspaceSize = (size_t *)((*cudnnlist)[4]);


    int mycomputesapce=(int) *IndicesSize+(int) *WorkspaceSize;
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       printf("%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }
    cudaError_t e = cudaMalloc((void**)&indices, *IndicesSize);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *IndicesSize;

 
    return 0;
    }


    e = cudaMalloc((void**)&workspace, *WorkspaceSize);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *WorkspaceSize;
      cudaFree(indices);
   
    return 0;
    }

    auto alpha = 1.0f, beta = 0.0f;


    CUDNN_CALL(cudnnReduceTensor(
        handle,
        reduceTensorDesc,
        indices,
        *IndicesSize,
        workspace,
        *WorkspaceSize,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));


    cudaFree(indices);
    cudaFree(workspace);
    //CUDNN_CALL(cudnnDestroy(handle));
   
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;



}



int DLGpuReduceSumAll(const DLArrayHandle input, DLArrayHandle output) {

    assert(1 == output->ndim);
    assert(1 == output->shape[0]);
    int stride = 1;
    int reduceDim = input->shape[0];
    int outputCount = 1;
    int lowstride = 1;
    for (int i = 0; i < input->ndim; ++i) {
        outputCount *= input->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;

    float* tmp = (float*)malloc(sizeof(float)*outputCount);
    float* tmp1 = (float*)malloc(sizeof(float)*outputCount);
    float* inputi = inputArr;
    float* outputi = tmp;
    float* changetmp;
    for(int i= (input->ndim)-1; i > 0;--i){
        outputCount /= input->shape[i];
        reduceDim = input->shape[i];
        lowstride = reduceDim * stride;

        matrix_reduce_sum_axis_n_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
            inputi, outputCount, outputi, reduceDim, stride, lowstride);

        if (i == (input->ndim)-1){
        changetmp = tmp1;
        }else{
        changetmp = inputi;
        }
        inputi = outputi;
        outputi = changetmp;
        
    }

    outputi = outputArr;
    lowstride = reduceDim * stride;
    matrix_reduce_sum_axis_n_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
        inputi, outputCount, outputi, reduceDim, stride, lowstride);
    free(tmp);
    free(tmp1);

    cudaDeviceSynchronize();
    return 0;
}

int DLGpuReduceSumAllBackward(const DLArrayHandle input, DLArrayHandle output) {

    assert(1 == input->ndim);
    assert(1 == input->shape[0]);
    float *val = (float*)malloc(sizeof(float));
    float* inputArr = (float*) input->data;
    cudaMemcpy(val, inputArr, sizeof(float), cudaMemcpyDeviceToHost);


  //  DLGpuArraySet(output, *val);
    cudaDeviceSynchronize();
    return 0;
}




int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim + 1);
    int zeroDim = input->shape[0], outputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {
        assert(input->shape[i+1] == output->shape[i]);
        outputCount *= output->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_reduce_sum_axis_zero_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
            inputArr, outputCount, outputArr, zeroDim);
    cudaDeviceSynchronize();
    return 0;
}



__global__ void matrix_concat_to_kernel(int count1,const float* inputArr1,int count2,const float* inputArr2,
                                        int outputCount, float* outputArr,int  count) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      int index1,index2;
      index1=index%count;
      index2=index/count;
      if(index1<count1)
        outputArr[index]=inputArr1[index2*count1+index1];
      else
        outputArr[index]=inputArr2[index2*count2+index1-count1];
  }
}


int DLGpuConcatForward(const DLArrayHandle input1, const DLArrayHandle input2,DLArrayHandle output, void **cudaStream) {
  assert(input1->ndim == input2->ndim);
  int inputCount1 = 1,inputCount2 = 1;
  int Count,count,OutputCount,Count1,Count2;
  for (int i = 0; i < input1->ndim; ++i) {
      if(i!=1)
        assert(input1->shape[i] == input2->shape[i]);
      inputCount1 *= input1->shape[i];
      inputCount2 *= input2->shape[i];
  }
    Count=input1->shape[0];
    OutputCount=inputCount1+inputCount2;
    count=OutputCount/Count;
    Count1=inputCount1/Count;
    Count2=inputCount2/Count;
    const float* input_data_a = (const float*)input1->data;
    const float* input_data_b = (const float*)input2->data;
    float* output_data = (float*)output->data;
    matrix_concat_to_kernel<<<BLOCK_NUM(OutputCount), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
     Count1,input_data_a,Count2, input_data_b,OutputCount,output_data,count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}


__global__ void matrix_concat_a_backward_to_kernel(int count1,const float* inputArr1,int count2, const float* inputArr2,
                                        int outputCount,const float* doutputArr,int  count,
                                        float* dinputArr1) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      int index1,index2;
      index1=index%count;
      index2=index/count;
      if(index1<count1)
        dinputArr1[index2*count1+index1]=doutputArr[index];
  }
}


int DLGpuConcataBackward(const DLArrayHandle input1,const DLArrayHandle input2,const DLArrayHandle doutput,
                        DLArrayHandle dinput1, void **cudaStream) {
  assert(input1->ndim == input2->ndim);
  int inputCount1 = 1,inputCount2 = 1;
  int Count,count,OutputCount,Count1,Count2;
  for (int i = 0; i < input1->ndim; ++i) {
      if(i!=1)
        assert(input1->shape[i] == input2->shape[i]);
      inputCount1 *= input1->shape[i];
      inputCount2 *= input2->shape[i];
  }
    Count=input1->shape[0];
    OutputCount=inputCount1+inputCount2;
    count=OutputCount/Count;
    Count1=inputCount1/Count;
    Count2=inputCount2/Count;

    const float* input_data_a = (const float*)input1->data;
    const float* input_data_b = (const float*)input2->data;
    const float* doutput_data = (float*)doutput->data;
    float* dinput_data_a = (float*)dinput1->data;
    matrix_concat_a_backward_to_kernel<<<BLOCK_NUM(OutputCount), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
     Count1,input_data_a,Count2, input_data_b,OutputCount,doutput_data,count,dinput_data_a);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

__global__ void matrix_concat_b_backward_to_kernel(int count1,const float* inputArr1,int count2, const float* inputArr2,
                                        int outputCount,const float* doutputArr,int  count,
                                        float* dinputArr2) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      int index1,index2;
      index1=index%count;
      index2=index/count;
      if(index1>=count1)
        dinputArr2[index2*count2+index1-count1]=doutputArr[index];
  }
}


int DLGpuConcatbBackward(const DLArrayHandle input1,const DLArrayHandle input2,const DLArrayHandle doutput,
                        DLArrayHandle dinput2, void **cudaStream) {
  assert(input1->ndim == input2->ndim);
  int inputCount1 = 1,inputCount2 = 1;
  int Count,count,OutputCount,Count1,Count2;
  for (int i = 0; i < input1->ndim; ++i) {
      if(i!=1)
        assert(input1->shape[i] == input2->shape[i]);
      inputCount1 *= input1->shape[i];
      inputCount2 *= input2->shape[i];
  }
    Count=input1->shape[0];
    OutputCount=inputCount1+inputCount2;
    count=OutputCount/Count;
    Count1=inputCount1/Count;
    Count2=inputCount2/Count;

    const float* input_data_a = (const float*)input1->data;
    const float* input_data_b = (const float*)input2->data;
    const float* doutput_data = (float*)doutput->data;
    float* dinput_data_b = (float*)dinput2->data;
    matrix_concat_b_backward_to_kernel<<<BLOCK_NUM(OutputCount), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
     Count1,input_data_a,Count2, input_data_b,OutputCount,doutput_data,count,dinput_data_b);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}




//can ndarray
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output, void **cudaStream) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_add_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
          matAData, matBData, outputData, count);
  cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output, void **cudaStream) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
  }


  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_add_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
          inputArr, val, outputArr, count);
  cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output, void **cudaStream) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_multiply_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
          matAData, matBData, outputData, count);
  cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output, void **cudaStream) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);

    count *= input->shape[i];
  }


  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_multipy_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
          inputArr, val, outputArr, count);
  cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC,void **cublasHandle, void **cudaStream) {
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  assert(matA->shape[transposeA ? 0 : 1] == matB->shape[transposeB ? 1 : 0]);
  assert(matA->shape[transposeA ? 1 : 0] == matC->shape[0]);
  assert(matB->shape[transposeB ? 0 : 1] == matC->shape[1]);

  cublasHandle_t handle = (cublasHandle_t)(*cublasHandle);

  const float* matAData = (const float*) matA->data;
  const float* matBData = (const float*) matB->data;
  float* matCData = (float*) matC->data;
  float alpha = 1, beta = 0;

  cublasSgemm(handle,
              (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeB ? matB->shape[0] : matB->shape[1]),
              (transposeA ? matA->shape[1] : matA->shape[0]),
              (transposeB ? matB->shape[1] : matB->shape[0]),
              &alpha,
              matBData, matB->shape[1],
matAData, matA->shape[1],
& beta,
matCData, (transposeB ? matB->shape[0] : matB->shape[1]));

cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        assert(input->shape[i] == output->shape[i]);
        count *= input->shape[i];
    }
    float* inputArr = (float*)input->data;
    float* outputArr = (float*)output->data;
    matrix_relu_kernel << <BLOCK_NUM(count), MAX_THREADS_NUM >> > (
        inputArr, outputArr, count);
    cudaDeviceSynchronize();
    return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
    DLArrayHandle output) {
    assert(input->ndim == in_grad->ndim);
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        assert(input->shape[i] == in_grad->shape[i]);
        assert(input->shape[i] == output->shape[i]);
        count *= input->shape[i];
    }
    const float* inputArr = (const float*)input->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    matrix_relu_gradient_kernel << <BLOCK_NUM(count), MAX_THREADS_NUM >> > (
        inputArr, gradArr, outputArr, count);
    cudaDeviceSynchronize();
    return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output, void **cudaStream) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0]);
    assert(input->shape[1] == output->shape[1]);

    int nRow = input->shape[0];
    int nCol = input->shape[1];

    dim3 block(MAX_THREADS_NUM);
    dim3 grid((nRow + block.x - 1) / block.x);

    float* inputArr = (float*)input->data;
    float* outputArr = (float*)output->data;

    matrix_softmax_kernel << <grid, block, 0, (cudaStream_t)*cudaStream >> > (nRow, nCol, inputArr, outputArr);

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
    const DLArrayHandle input_b,
    DLArrayHandle output, void **cudaStream) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    const float* input_data_b = (const float*)input_b->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel << <1, threads, nrow * sizeof(float), (cudaStream_t)*cudaStream >> > (
        nrow, ncol, input_data_a, input_data_b, output_data);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuMatrixExp(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_exp_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    inputArr, outputArr, count);
    cudaDeviceSynchronize();
    return 0;
}


int DLGpuMatrixLog(const DLArrayHandle input, DLArrayHandle output, void **cudaStream) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_log_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    inputArr, outputArr, count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}


int DLGpuMatrixReverse(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_reverse_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    inputArr, outputArr, count);
    cudaDeviceSynchronize();
    return 0;
}

int DLGpuMatrixPow(const DLArrayHandle input,const float val, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
    }
    float* inputArr = (float*) input->data;
    float* outputArr = (float*) output->data;
    matrix_pow_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    inputArr, val, outputArr, count);
    cudaDeviceSynchronize();
    return 0;
}


int DLGpuCreatecudaStream(void **cudaStream){

    cudaStream_t handle;

    cudaStreamCreate(&handle);



    *cudaStream = handle;

    return 0;

}

int DLGpuDestroycudaStream(void **cudaStream){


    cudaStream_t handle = (cudaStream_t)(*cudaStream);

    cudaStreamDestroy(handle);

    return 0;

}


int DLGpuCreatecudnnHandle(void **cudnnHandle, void **cudaStream){


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));
    CUDNN_CALL(cudnnSetStream(handle,(cudaStream_t)(*cudaStream)));

    *cudnnHandle = handle;

    return 0;

}

int DLGpuDestroycudnnHandle(void **cudnnHandle){


    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);
    CUDNN_CALL(cudnnDestroy(handle));


    return 0;

}



int DLGpuCreatecublasHandle(void **cublasHandle, void **cudaStream){


    //handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle,(cudaStream_t)(*cudaStream));


    *cublasHandle = handle;

    return 0;

}

int DLGpuDestroycublasHandle(void **cublasHandle){


    cublasHandle_t handle = (cublasHandle_t)(*cublasHandle);
    cublasDestroy(handle);


    return 0;

}




//3ά
int DLGpuConvolution1DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream) {

    //cout<<dataformat<<endl;
   // cout<<padding<<endl;
    assert(input->ndim == 3);
    assert(filter->ndim == 3);

    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);


    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;

    cudaError_t e = cudaMalloc(&workspace, workspace_size);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) workspace_size;

  
    return 0;
    }


    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));
    //�ڴ�
    cudaFree(workspace);
    //CUDNN_CALL(cudnnDestroy(handle));

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;

}


int DLGpuConvolution1DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int v,
    void ***cudnnlist) {

   // cout << dataformat << endl;
   // cout << padding << endl;


    int input_n = input_shapes[0];
    int input_c = input_shapes[2];
    int input_h = 1;
    int input_w = input_shapes[1];


    int filter_n = filter_shapes[0];
    int filter_c = filter_shapes[2];
    int filter_h = 1;
    int filter_w = filter_shapes[1];

    if (dataformat == 0) {
        input_n = input_shapes[0];
        input_c = input_shapes[1];
        input_h = 1;
        input_w = input_shapes[2];

        filter_n = filter_shapes[0];
        filter_c = filter_shapes[1];
        filter_h = 1;
        filter_w = filter_shapes[2];
    }


    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    int u = 1;

    if (padding == 1) {
        pad_w = filter_w / 2;
    }



    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));



    if (dataformat == 0) {
        output_shapes[0] = out_n;
        output_shapes[1] = out_c;
        output_shapes[2] = out_w;
    }
    else {
        output_shapes[0] = out_n;
        output_shapes[1] = out_w;
        output_shapes[2] = out_c;
    }

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

    * cudnnlist = (void **)malloc(sizeof(void *)*4);
    (* cudnnlist)[0] = input_descriptor;
    (* cudnnlist)[1] = filter_descriptor;
    (* cudnnlist)[2] = conv_descriptor;
    (* cudnnlist)[3] = output_descriptor;

    
    return 0;

}

int DLGpuConvolutionBackwardFilter(const DLArrayHandle input,
    const DLArrayHandle doutput,
    const DLArrayHandle filter,
    DLArrayHandle dfilter,
    void*** cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace) {

    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);



    //�������㷨
    cudnnConvolutionBwdFilterAlgo_t  algo1;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &algo1));


    size_t workspace_size1 = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        algo1,
        &workspace_size1));
    void* workspace1 = nullptr;
    int mycomputesapce=(int) workspace_size1;
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       // printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }

    cudaError_t e = cudaMalloc(&workspace1, workspace_size1);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) workspace_size1;

    
    return 0;
    }

    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo1,
        workspace1,
        workspace_size1,
        &beta, //y����,y�������ݽ������ţ�
        filter_descriptor,
        dfilter->data));



    //�ڴ�
    cudaFree(workspace1);
    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuConvolutionBackwardData(const DLArrayHandle input,
    const DLArrayHandle doutput,
    const DLArrayHandle filter,
    DLArrayHandle dinput,
    void*** cudnnlist,
    void ** cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace) {



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);



    cudnnConvolutionBwdDataAlgo_t algo2;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo2));



    size_t workspace_size2 = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        algo2,
        &workspace_size2));
    void* workspace2 = nullptr;
    int mycomputesapce=(int) workspace_size2;
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       // printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }

    cudaError_t e = cudaMalloc(&workspace2, workspace_size2);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) workspace_size2;


    return 0;
    }


    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
        &alpha, //x*w����
        filter_descriptor,
        filter->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo2,
        workspace2,
        workspace_size2,
        &beta, //y����,y�������ݽ������ţ�
        input_descriptor,
        dinput->data));


    cudaFree(workspace2);
    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;

}


// int DLGpuConvolution1DBackward(const DLArrayHandle input,
//     const DLArrayHandle doutput,
//     const DLArrayHandle filter,
//     DLArrayHandle dfilter,
//     DLArrayHandle dinput,
//     void ***cudnnlist,
//     void ** cudnnHandle) {

//     assert(input->ndim == 3);
//     assert(filter->ndim == 3);


//     //handle
//     //cudnnHandle_t handle;
//     //CUDNN_CALL(cudnnCreate(&handle));
//     cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    
//     //input
//     cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

//     //filter
//     cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


//     //��ķ�ʽ��������padding
//     cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

//     //output��Ϣ
//     cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);
    

    
//     //�������㷨
//     cudnnConvolutionBwdFilterAlgo_t  algo1;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
//         0,
//         &algo1));

//     cudnnConvolutionBwdDataAlgo_t algo2;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
//         0,
//         &algo2));

//     //׼����������Ŀռ�


//     size_t workspace_size1= 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         algo1,
//         &workspace_size1));
//     void* workspace1= nullptr;
//     cudaMalloc(&workspace1, workspace_size1);

//     size_t workspace_size2 = 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         algo2,
//         &workspace_size2));
//     void* workspace2 = nullptr;
//     cudaMalloc(&workspace2, workspace_size2);


//     // convolution
//     auto alpha = 1.0f, beta = 0.0f;
//     CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
//         &alpha, //x*w����
//         input_descriptor,
//         input->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo1,
//         workspace1,
//         workspace_size1,
//         &beta, //y����,y�������ݽ������ţ�
//         filter_descriptor,
//         dfilter->data));


//     CUDNN_CALL(cudnnConvolutionBackwardData(handle,
//         &alpha, //x*w����
//         filter_descriptor,
//         filter->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo2,
//         workspace2,
//         workspace_size2,
//         &beta, //y����,y�������ݽ������ţ�
//         input_descriptor,
//         dinput->data));


//     //�ڴ�
//     cudaFree(workspace1);
//     cudaFree(workspace2);
//     //CUDNN_CALL(cudnnDestroy(handle));
//     cudaDeviceSynchronize();
   // return 0;

// }





int DLGpuConvolution2DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    void ***cudnnlist,
    void ** cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace) {

    assert(input->ndim == 4);
    assert(filter->ndim == 4);


    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);



    //filter
    cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);




    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;

    int mycomputesapce=(int) workspace_size;
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       // printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }
    cudaError_t e = cudaMalloc(&workspace, workspace_size);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) workspace_size;

 
    return 0;
    }

    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));

    //�ڴ�
    cudaFree(workspace);
    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuConvolution2DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int u,          /* vertical filter stride */
    const int v,
    void ***cudnnlist) {
   // cout << dataformat << endl;
   // cout << padding << endl;


    int input_n = input_shapes[0];
    int input_c = input_shapes[3];
    int input_h = input_shapes[1];
    int input_w = input_shapes[2];


    int filter_n = filter_shapes[0];
    int filter_c = filter_shapes[3];
    int filter_h = filter_shapes[1];
    int filter_w = filter_shapes[2];

    if (dataformat == 0) {
        input_n = input_shapes[0];
        input_c = input_shapes[1];
        input_h = input_shapes[2];
        input_w = input_shapes[3];

        filter_n = filter_shapes[0];
        filter_c = filter_shapes[1];
        filter_h = filter_shapes[2];
        filter_w = filter_shapes[3];
    }


    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    if (padding == 1) {
        pad_h = filter_h / 2;
        pad_w = filter_w / 2;
    }


    // //handle
    // cudnnHandle_t handle;
    // CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));



    if (dataformat == 0) {
        output_shapes[0] = out_n;
        output_shapes[1] = out_c;
        output_shapes[2] = out_h;
        output_shapes[3] = out_w;
    }
    else {
        output_shapes[0] = out_n;
        output_shapes[1] = out_h;
        output_shapes[2] = out_w;
        output_shapes[3] = out_c;
    }

    

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

     * cudnnlist = (void **)malloc(sizeof(void *)*4);
    (* cudnnlist)[0] = input_descriptor;
    (* cudnnlist)[1] = filter_descriptor;
    (* cudnnlist)[2] = conv_descriptor;
    (* cudnnlist)[3] = output_descriptor;



 
    return 0;

}

// int Test(cudnnTensorDescriptor_t i){


//     printf("%p\n",i);
//     cudnnDataType_t *dataType = (cudnnDataType_t *)malloc(sizeof(cudnnDataType_t *));

//     int                     n=0;
//         int                     c=0;
//         int                     h=0;
//         int                     w=0;
//         int                     nStride=0;
//         int                     cStride=0;
//         int                     hStride=0;
//         int                     wStride=0;
//     cudnnGetTensor4dDescriptor(
//         i,
//         dataType,
//         &n,
//         &c,
//                             &h,
//                           &w,
//                              &nStride,
//                              &cStride,
//                             &hStride,
//                              &wStride);

//     printf("%d\n",n);
//     printf("%d\n",c);
//     printf("%d\n",h);
//     printf("%d\n",w);

    



//     cudaDeviceSynchronize();
  //  return 0;


// }



// int DLGpuConvolution2DBackward(const DLArrayHandle input,
//     const DLArrayHandle doutput,
//     const DLArrayHandle filter,
//     DLArrayHandle dfilter,
//     DLArrayHandle dinput,
//     void ***cudnnlist,
//     void **cudnnHandle) {

//     assert(input->ndim == 4);
//     assert(filter->ndim == 4);

   


//     //handle
//     //cudnnHandle_t handle;
//     //CUDNN_CALL(cudnnCreate(&handle));
//     cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

//     cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

//     //filter
//     cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


//     //��ķ�ʽ��������padding
//     cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

//     //output��Ϣ
//     cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);

//     cudnnConvolutionBwdFilterAlgo_t  algo1;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
//         0,
//         &algo1));

//     cudnnConvolutionBwdDataAlgo_t algo2;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
//         0,
//         &algo2));

//     //׼����������Ŀռ�


//     size_t workspace_size1= 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         algo1,
//         &workspace_size1));
//     void* workspace1= nullptr;
//     cudaMalloc(&workspace1, workspace_size1);

//     size_t workspace_size2 = 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         algo2,
//         &workspace_size2));
//     void* workspace2 = nullptr;
//     cudaMalloc(&workspace2, workspace_size2);



//     // convolution
//     auto alpha = 1.0f, beta = 0.0f;
//     CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
//         &alpha, //x*w����
//         input_descriptor,
//         input->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo1,
//         workspace1,
//         workspace_size1,
//         &beta, //y����,y�������ݽ������ţ�
//         filter_descriptor,
//         dfilter->data));



//     CUDNN_CALL(cudnnConvolutionBackwardData(handle,
//         &alpha, //x*w����
//         filter_descriptor,
//         filter->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo2,
//         workspace2,
//         workspace_size2,
//         &beta, //y����,y�������ݽ������ţ�
//         input_descriptor,
//         dinput->data));


//     //�ڴ�
//     cudaFree(workspace1);
//     cudaFree(workspace2);
//     //CUDNN_CALL(cudnnDestroy(handle));
//     cudaDeviceSynchronize();
  //  return 0;
// }

//5ά
int DLGpuConvolution3DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    void ***cudnnlist,
    void ** cudnnHandle, int *memorytoSaving) {

    assert(input->ndim == 5);
    assert(filter->ndim == 5);

    

    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);

    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;

    cudaError_t e = cudaMalloc(&workspace, workspace_size);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) workspace_size;

     
    return 0;
    }


    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));

    //�ڴ�
    cudaFree(workspace);
    //CUDNN_CALL(cudnnDestroy(handle));


   cudaDeviceSynchronize();
    return 0;




}


int DLGpuConvolution3DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int s1,
    const int s2,
    const int s3,
    void ***cudnnlist) {

    int* padA, * filterStrideA, * dilationA;

    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    dilationA = (int*)malloc(sizeof(int) * 3);


    for (int i = 0; i < 3; i++) {
        padA[i] = 0;
        dilationA[i] = 1;

    }

    if (padding == 1) {
        for (int i = 0; i < 3; i++) {
            padA[i] = filter_shapes[i+2]/2;

        }
    }

    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;


    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shapes));

    //�˺���
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        5,
        filter_shapes));


    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_descriptor,
        3,
        padA,
        filterStrideA,
        dilationA,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //output��shape
    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        5,
        output_shapes));



    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shapes));

     * cudnnlist = (void **)malloc(sizeof(void *)*4);
    (* cudnnlist)[0] = input_descriptor;
    (* cudnnlist)[1] = filter_descriptor;
    (* cudnnlist)[2] = conv_descriptor;
    (* cudnnlist)[3] = output_descriptor;


    return 0;

}

// int DLGpuConvolution3DBackward(const DLArrayHandle input,
//     const DLArrayHandle doutput,
//     const DLArrayHandle filter,
//     DLArrayHandle dfilter,
//     DLArrayHandle dinput,
//     void ***cudnnlist,
//     void ** cudnnHandle) {

//     assert(input->ndim == 5);
//     assert(filter->ndim == 5);

    

//      //cudnnHandle_t handle;
//     //CUDNN_CALL(cudnnCreate(&handle));
//     cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


//     cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

//     //filter
//     cudnnFilterDescriptor_t filter_descriptor = (cudnnFilterDescriptor_t)((*cudnnlist)[1]);


//     //��ķ�ʽ��������padding
//     cudnnConvolutionDescriptor_t conv_descriptor = (cudnnConvolutionDescriptor_t)((*cudnnlist)[2]);

//     //output��Ϣ
//     cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[3]);

//     cudnnConvolutionBwdFilterAlgo_t  algo1;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
//         0,
//         &algo1));

//     cudnnConvolutionBwdDataAlgo_t algo2;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
//         0,
//         &algo2));

//     //׼����������Ŀռ�


//     size_t workspace_size1= 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
//         input_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         filter_descriptor,
//         algo1,
//         &workspace_size1));
//     void* workspace1= nullptr;
//     cudaMalloc(&workspace1, workspace_size1);

//     size_t workspace_size2 = 0;
//     CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
//         filter_descriptor,
//         output_descriptor,
//         conv_descriptor,
//         input_descriptor,
//         algo2,
//         &workspace_size2));
//     void* workspace2 = nullptr;
//     cudaMalloc(&workspace2, workspace_size2);



//     // convolution
//     auto alpha = 1.0f, beta = 0.0f;
//     CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
//         &alpha, //x*w����
//         input_descriptor,
//         input->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo1,
//         workspace1,
//         workspace_size1,
//         &beta, //y����,y�������ݽ������ţ�
//         filter_descriptor,
//         dfilter->data));



//     CUDNN_CALL(cudnnConvolutionBackwardData(handle,
//         &alpha, //x*w����
//         filter_descriptor,
//         filter->data,
//         output_descriptor,
//         doutput->data,
//         conv_descriptor,
//         algo2,
//         workspace2,
//         workspace_size2,
//         &beta, //y����,y�������ݽ������ţ�
//         input_descriptor,
//         dinput->data));


//     //�ڴ�
//     cudaFree(workspace1);
//     cudaFree(workspace2);
//     //CUDNN_CALL(cudnnDestroy(handle));
//     cudaDeviceSynchronize();
 //   return 0;
// }



int DLGpuPooling1DForward(const DLArrayHandle input,
    DLArrayHandle output,
    void ***cudnnlist,
    void ** cudnnHandle) {




    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);



    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));

    //CUDNN_CALL(cudnnDestroy(handle));
    cudaDeviceSynchronize();
    return 0;
}


int DLGpuPooling1DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding_w,
    const int v,
    const int filter_w,
    void ***cudnnlist){



    int padding_h = 0;
    int u = 1;
    int filter_h = 1;

    int input_n = input_shapes[0];
    int input_c = input_shapes[2];
    int input_h = 1;
    int input_w = input_shapes[1];



    if (dataformat == 0) {
        input_n = input_shapes[0];
        input_c = input_shapes[1];
        input_h = 1;
        input_w = input_shapes[2];
    }

    int output_n;
    int output_c;
    int output_h;
    int output_w;



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        poolingMode,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));


    if (dataformat == 0) {
        output_shapes[0] = output_n;
        output_shapes[1] = output_c;
        output_shapes[2] = output_w;
    }
    else {
        output_shapes[0] = output_n;
        output_shapes[1] = output_w;
        output_shapes[2] = output_c;
    }

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));

     *cudnnlist = (void **)malloc(sizeof(void *)*3);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = pool_descriptor;
    (*cudnnlist)[2] = output_descriptor;

    return 0;


}

int DLGpuPooling1DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void ***cudnnlist,
    void **cudnnHandle) {




    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);


    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));


    //CUDNN_CALL(cudnnDestroy(handle));
    cudaDeviceSynchronize();
    return 0;
}



int DLGpuPooling2DForward(const DLArrayHandle input,
    DLArrayHandle output,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream)
{



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    // printf("%p",cudnnlist);
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);
    // printf("dddddd");
    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);

    // printf("dddddd");
    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);

    // printf("dddddd");

    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));


    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}

int DLGpuPooling2DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding_h,
    const int padding_w,
    const int u,
    const int v,
    const int filter_h,
    const int filter_w,
    void ***cudnnlist){

    //printf("DLGpuPooling2DForwardGetOutShape\n");

    int input_n = input_shapes[0];
    int input_c = input_shapes[3];
    int input_h = input_shapes[1];
    int input_w = input_shapes[2];
    if (dataformat == 0) {
        input_n = input_shapes[0];
        input_c = input_shapes[1];
        input_h = input_shapes[2];
        input_w = input_shapes[3];
    }



    int output_n;
    int output_c;
    int output_h;
    int output_w;



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        poolingMode,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));


    if (dataformat == 0) {
        output_shapes[0] = output_n;
        output_shapes[1] = output_c;
        output_shapes[2] = output_h;
        output_shapes[3] = output_w;
    }
    else {
        output_shapes[0] = output_n;
        output_shapes[1] = output_h;
        output_shapes[2] = output_w;
        output_shapes[3] = output_c;
    }

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));

    *cudnnlist = (void **)malloc(sizeof(void *)*3);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = pool_descriptor;
    (*cudnnlist)[2] = output_descriptor;

  
    return 0;


}

int DLGpuPooling2DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream)
{

    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

   cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);

    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));


    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}




int DLGpuPooling3DForward(const DLArrayHandle input,
    DLArrayHandle output,
    void ***cudnnlist,
    void **cudnnHandle)
{
    assert(input->ndim == 5);



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);



    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));


    //CUDNN_CALL(cudnnDestroy(handle));
    cudaDeviceSynchronize();
    return 0;

}

int DLGpuPooling3DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding1,
    const int padding2,
    const int padding3,
    const int s1,
    const int s2,
    const int s3,
    const int filter1,
    const int filter2,
    const int filter3,
    void ***cudnnlist){

    int* filter_shape,*padA,*filterStrideA;

    filter_shape = (int*)malloc(sizeof(int) * 3);
    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    filter_shape[0] = filter1;
    filter_shape[1] = filter2;
    filter_shape[2] = filter3;
    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;
    padA[0] = padding1;
    padA[1] = padding2;
    padA[2] = padding3;



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));


     cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(pool_descriptor,
        poolingMode,
        CUDNN_NOT_PROPAGATE_NAN,
        3,
        filter_shape,
        padA,
        filterStrideA));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shapes));


    CUDNN_CALL(cudnnGetPoolingNdForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        5,
        output_shapes));

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shapes));

    *cudnnlist = (void **)malloc(sizeof(void *)*3);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = pool_descriptor;
    (*cudnnlist)[2] = output_descriptor;

   
    return 0;


}


int DLGpuPooling3DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void ***cudnnlist,
    void **cudnnHandle)
{
    assert(input->ndim == 5);




    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);


   cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnPoolingDescriptor_t pool_descriptor = (cudnnPoolingDescriptor_t)((*cudnnlist)[1]);


    //��ķ�ʽ��������padding
    cudnnTensorDescriptor_t output_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[2]);


    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));


    //CUDNN_CALL(cudnnDestroy(handle));
    cudaDeviceSynchronize();
    return 0;

}






//activation
//activation
int DLGpuActivationForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream) {



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);
    

    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);




    auto alpha = 1.0f, beta = 0.0f;
    if (activationMode != 3 ){
        // 描述激活
        cudnnActivationDescriptor_t activation_descriptor = (cudnnActivationDescriptor_t)((*cudnnlist)[1]);



        CUDNN_CALL(cudnnActivationForward(handle,
            activation_descriptor,
            &alpha,
            input_descriptor,
            input->data,
            &beta,
            input_descriptor,
            output->data));




    }else{

        CUDNN_CALL(cudnnSoftmaxForward(handle,
            CUDNN_SOFTMAX_FAST,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            input_descriptor,
            input->data,
            &beta,
            input_descriptor,
            output->data));


    }




    //CUDNN_CALL(cudnnDestroy(handle));

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;



}

int DLGpuGetInputDescriptor(const int *input_shapes,
    const int sizeofshape,
    cudnnTensorFormat_t dataformat,
    void ***inputd){
    
    cudnnTensorDescriptor_t input_descriptor;

    int input_n;
    int input_c;
    int input_h;
    int input_w;

    if(sizeofshape == 3){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = 1;
            input_w = input_shapes[2];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[2];
            input_h = 1;
            input_w = input_shapes[1];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }


    if(sizeofshape == 4){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = input_shapes[2];
            input_w = input_shapes[3];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[3];
            input_h = input_shapes[1];
            input_w = input_shapes[2];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }

    if(sizeofshape == 5){

        int* input_shape;
        input_shape = (int*)malloc(sizeof(int) * 5);
        for(int i=0;i<5;i++){
            input_shape[i]= input_shapes[i];
        }

        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            5,
            input_shape));



    }

    * inputd = (void **)malloc(sizeof(void *)*1);
    (*inputd)[0] = input_descriptor;

    return 0;
}


int DLGpuActivationGetCudnnlist(const int *input_shapes,
    const int sizeofshape,
    cudnnTensorFormat_t dataformat,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist){


        //input
    cudnnTensorDescriptor_t input_descriptor;

    int input_n;
    int input_c;
    int input_h;
    int input_w;

    if(sizeofshape == 3){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = 1;
            input_w = input_shapes[2];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[2];
            input_h = 1;
            input_w = input_shapes[1];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }


    if(sizeofshape == 4){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = input_shapes[2];
            input_w = input_shapes[3];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[3];
            input_h = input_shapes[1];
            input_w = input_shapes[2];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }

    if(sizeofshape == 5){

        int* input_shape;
        input_shape = (int*)malloc(sizeof(int) * 5);
        for(int i=0;i<5;i++){
            input_shape[i]= input_shapes[i];
        }

        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            5,
            input_shape));



    }

    * cudnnlist = (void **)malloc(sizeof(void *)*2);
    (*cudnnlist)[0] = input_descriptor;


  
    if (activationMode != 3 ){
        // 描述激活
        cudnnActivationDescriptor_t activation_descriptor;
        CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_descriptor));
        CUDNN_CALL(cudnnSetActivationDescriptor(activation_descriptor,
            activationMode,
            CUDNN_PROPAGATE_NAN,
            /*relu_coef=*/0));

        (*cudnnlist)[1] = activation_descriptor;


    }

   
    return 0;


}


//activation
int DLGpuActivationBackward(const DLArrayHandle input,
    DLArrayHandle dinput,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream) {

    //assert(input->ndim==4||input->ndim==3||input->ndim==5);

    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);





    auto alpha = 1.0f, beta = 0.0f;
    if(activationMode != 3){

        // 描述激活
        cudnnActivationDescriptor_t activation_descriptor = (cudnnActivationDescriptor_t)((*cudnnlist)[1]);

        // 激活函数求导

        CUDNN_CALL(cudnnActivationBackward(handle,
            activation_descriptor,
            &alpha,
            input_descriptor,
            output->data,
            input_descriptor,
            doutput->data,
            input_descriptor,
            input->data,
            &beta,
            input_descriptor,
            dinput->data));

    }else{


        CUDNN_CALL(cudnnSoftmaxBackward(handle,
            CUDNN_SOFTMAX_FAST,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            input_descriptor,
            output->data,
            input_descriptor,
            doutput->data,
            &beta,
            input_descriptor,
            dinput->data));


    }


    //CUDNN_CALL(cudnnDestroy(handle));

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;



}

//dropout
int DLGpuDropoutForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const float dropout,
    const int seed,
    void **reserveSpace_p,/*back use*/
    void ***inputd,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace){



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*inputd)[0]);;

    
    size_t stateSizeInBytes = 1;
    size_t reserveSpaceSizeInBytes = 1;
    void *states;
    //unsigned long long  seed = 0;//ini rand seed
    CUDNN_CALL(cudnnDropoutGetStatesSize(handle,
        &stateSizeInBytes));

    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(input_descriptor,
        &reserveSpaceSizeInBytes));


    int mycomputesapce=(int) stateSizeInBytes+(int) reserveSpaceSizeInBytes;
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       // printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }
    cudaError_t e = cudaMalloc((void**)&states, stateSizeInBytes);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) stateSizeInBytes;

      
    return 0;
    }

    e = cudaMalloc((void**)reserveSpace_p, reserveSpaceSizeInBytes);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) reserveSpaceSizeInBytes;
      cudaFree(states);
     
    return 0;
    }



    cudnnDropoutDescriptor_t dropout_descriptor;
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_descriptor));
    //printf("before\n");
    cudnnStatus_t err=cudnnSetDropoutDescriptor(dropout_descriptor,
        handle,
        dropout,
        states,
        stateSizeInBytes,
        seed);
     if(err != CUDNN_STATUS_SUCCESS){\
         //printf("出错了");
        *memorytoSaving = 100*pow(2,20);

          cudaFree(states);
          cudaFree(*reserveSpace_p);
          return 0;
    }\
    if (err != CUDNN_STATUS_SUCCESS) {\
        \
            printf("出错了");
            std::cout << "    Error occurred: " << err << std::endl; \
            std::exit(1); \
    } \
   // printf("正确了");
   // printf("%d,err",err);
    //printf("after\n");

    CUDNN_CALL(cudnnDropoutForward(handle,
        dropout_descriptor,
        input_descriptor,
        input->data,
        input_descriptor,
        output->data,
        *reserveSpace_p,
        reserveSpaceSizeInBytes));




    //CUDNN_CALL(cudnnDestroy(handle));
    cudaFree(states);

    * cudnnlist = (void **)malloc(sizeof(void *)*2);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = dropout_descriptor;

    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;

}






int DLGpuDropoutBackward(const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void **reserveSpace_p,/*back use*/
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream){






    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);




    size_t reserveSpaceSizeInBytes = 1;


    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(input_descriptor,
        &reserveSpaceSizeInBytes));

    cudnnDropoutDescriptor_t dropout_descriptor = (cudnnDropoutDescriptor_t)((*cudnnlist)[1]);





    CUDNN_CALL(cudnnDropoutBackward(handle,
        dropout_descriptor,
        input_descriptor,
        doutput->data,
        input_descriptor,
        dinput->data,
        *reserveSpace_p,
        reserveSpaceSizeInBytes));


    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_descriptor));
    //CUDNN_CALL(cudnnDestroy(handle));
    cudaFree(*reserveSpace_p);
    free(*cudnnlist);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;

}




//loss

__global__ void matrix_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
   output[x+y*ncol] = input_b[x] * log(input_a[x])+(1-input_b[x])* log(1-input_a[x]);
  }
  loss_per_row[y] = loss;
  /*__syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }

    output[0] = mean_loss;
  }*/
}
__global__ void matrix_l1loss_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss += abs(input_b[x]-input_a[x]);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}
__global__ void matrix_l2loss_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss += pow(input_b[x]-input_a[x],2);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    mean_loss/=2;
    output[0] = mean_loss;
  }
}

__global__ void matrix_l1lossgradient_kernel(const float* inputArr,const float* inputArr1, const float* gradArr,
                                            float* outputArr, int count,int n) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = (inputArr[index]-inputArr1[index])> 0 ? gradArr[0]/n : -gradArr[0] /n;
    }
}
__global__ void matrix_l2lossgradient_kernel(const float* inputArr,const float* inputArr1, const float* gradArr,
                                            float* outputArr, int count,int n) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = (inputArr[index]-inputArr1[index])*gradArr[0]/n;
    }
}
__global__ void matrix_l1regular_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss += abs(input_a[x]);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}
__global__ void matrix_l2regular_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss += pow(input_a[x],2);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    mean_loss/=2;
    output[0] = mean_loss;
  }
}
__global__ void matrix_l1regulargradient_kernel(const float* inputArr, const float* gradArr,
                                            float* outputArr, int count,int n) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index]> 0 ? gradArr[0]/n : -gradArr[0]/n;
    }
}
__global__ void matrix_l2regulargradient_kernel(const float* inputArr, const float* gradArr,
                                            float* outputArr, int count,int n) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index]*gradArr[0]/n;
    }
}


int DLGpuCrossEntropy(const DLArrayHandle input_a,
    const DLArrayHandle input_b,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 2);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    const float* input_data_b = (const float*)input_b->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_cross_entropy_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, input_data_b, output_data);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL1loss(const DLArrayHandle input_a,
    const DLArrayHandle input_b,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    const float* input_data_b = (const float*)input_b->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_l1loss_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, input_data_b, output_data);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL2loss(const DLArrayHandle input_a,
    const DLArrayHandle input_b,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    const float* input_data_b = (const float*)input_b->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_l2loss_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, input_data_b, output_data);
    cudaDeviceSynchronize();
    return 0;
}

int DLGpuL1LossGradient(const DLArrayHandle input, const DLArrayHandle input1,const DLArrayHandle in_grad,
    DLArrayHandle output) {

    assert(input->ndim == input1->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->ndim == input1->ndim);
        count *= input->shape[i];
    }

    int  n=input->shape[0];
    const float* inputArr = (const float*)input->data;
    const float* inputArr1 = (const float*)input1->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    int nrow=input->shape[0];
    dim3 threads;
    threads.x = nrow;

    matrix_l1lossgradient_kernel << <1, threads >> > (
        inputArr, inputArr1,gradArr, outputArr, count,n);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL2LossGradient(const DLArrayHandle input, const DLArrayHandle input1,const DLArrayHandle in_grad,
    DLArrayHandle output) {

    assert(input->ndim == input1->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
    assert(input->ndim == input1->ndim);
        count *= input->shape[i];
    }

    int  n=input->shape[0];
    const float* inputArr = (const float*)input->data;
    const float* inputArr1 = (const float*)input1->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    int nrow=input->shape[0];
    dim3 threads;
    threads.x = nrow;

    matrix_l2lossgradient_kernel << <1, threads >> > (
        inputArr, inputArr1,gradArr, outputArr, count,n);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL1regular(const DLArrayHandle input_a,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(output->ndim == 1);
    int nrow = input_a->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_l1regular_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, output_data);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL2regular(const DLArrayHandle input_a,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(output->ndim == 1);
    int nrow = input_a->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_l2regular_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, output_data);
    cudaDeviceSynchronize();
    return 0;
}

int DLGpuL1regularGradient(const DLArrayHandle input,const DLArrayHandle in_grad,
    DLArrayHandle output) {

    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        count *= input->shape[i];
    }

    int  n=input->shape[0];
    const float* inputArr = (const float*)input->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    int nrow=input->shape[0];
    dim3 threads;
    threads.x = nrow;

    matrix_l1regulargradient_kernel << <1, threads >> > (
        inputArr, gradArr, outputArr, count,n);
    cudaDeviceSynchronize();
    return 0;
}
int DLGpuL2regularGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
    DLArrayHandle output) {

    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        count *= input->shape[i];
    }

    int  n=input->shape[0];
    const float* inputArr = (const float*)input->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    int nrow=input->shape[0];
    dim3 threads;
    threads.x = nrow;

    matrix_l2regulargradient_kernel << <1, threads >> > (
        inputArr, gradArr, outputArr, count,n);
    cudaDeviceSynchronize();
    return 0;
}




int DLGpuBatchNormalizationGetCudnnlist(const int *input_shapes,
    int sizeofshape,
    cudnnTensorFormat_t dataformat,
    cudnnBatchNormMode_t batchNormMode,
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist){

    //printf("DLGpuBatchNormalizationGetCudnnlist\n");

    cudnnTensorDescriptor_t input_descriptor;

    int input_n;
    int input_c;
    int input_h;
    int input_w;

    if(sizeofshape == 3){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = 1;
            input_w = input_shapes[2];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[2];
            input_h = 1;
            input_w = input_shapes[1];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }


    if(sizeofshape == 4){
        if (dataformat == 0) {
            input_n = input_shapes[0];
            input_c = input_shapes[1];
            input_h = input_shapes[2];
            input_w = input_shapes[3];
        }else{
            input_n = input_shapes[0];
            input_c = input_shapes[3];
            input_h = input_shapes[1];
            input_w = input_shapes[2];
        }
        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            input_n,
            input_c,
            input_h,
            input_w));

    }

    if(sizeofshape == 5){

        int* input_shape;
        input_shape = (int*)malloc(sizeof(int) * 5);
        for(int i=0;i<5;i++){
            input_shape[i]= input_shapes[i];
        }

        //input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
            dataformat,
            CUDNN_DATA_FLOAT,
            5,
            input_shape));



    }


    cudnnTensorDescriptor_t derivedBnDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&derivedBnDesc));
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(derivedBnDesc,
        input_descriptor,
        batchNormMode));


    size_t *s = (size_t *)malloc(sizeof(size_t));
    CUDNN_CALL(cudnnGetTensorSizeInBytes(derivedBnDesc,
        s));




    *cudnnlist = (void **)malloc(sizeof(void *)*3);
    (*cudnnlist)[0] = input_descriptor;
    (*cudnnlist)[1] = derivedBnDesc;
    (*cudnnlist)[2] = s;


    return 0;
}



//BatchNormalization
int DLGpuBatchNormalizationForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnBatchNormMode_t batchNormMode,
    int n,//第n+1次使用
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace) {



    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);
    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    cudnnTensorDescriptor_t derivedBnDesc = (cudnnTensorDescriptor_t)((*cudnnlist)[1]);

     size_t *s = (size_t *)((*cudnnlist)[2]);
     float *bnScalec = (float*)malloc(*s);
    float *bnBiasc = (float*)malloc(*s);
    for(int i=0;i< *s / sizeof(float);i++){
        bnScalec[i] = 1.0;
        bnBiasc[i] =  0.0;
    }
    float *bnScale,*bnBias;
    float *resultSaveMean;
    float *resultSaveInvVariance;


    int mycomputesapce=4*((int) *s);
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       //printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }
    cudaError_t e = cudaMalloc((void**)&bnScale, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;

      
    return 0;
    }
    e = cudaMalloc((void**)&bnBias, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;
      cudaFree(bnScale);
    
    return 0;
    }
    e = cudaMalloc((void**)&resultSaveMean, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;
      cudaFree(bnScale);
      cudaFree(bnBias);
    
    return 0;
    }
    e = cudaMalloc((void**)&resultSaveInvVariance, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;
      cudaFree(bnScale);
      cudaFree(bnBias);
      cudaFree(resultSaveMean);
     
    return 0;
    }




    cudaMemcpy(bnScale, bnScalec, *s, cudaMemcpyHostToDevice);
    cudaMemcpy(bnBias, bnBiasc, *s, cudaMemcpyHostToDevice);


    float *resultRunningMean = nullptr;
    float *resultRunningVariance = nullptr;




    *mean_p = resultSaveMean;
    *Variance_p = resultSaveInvVariance;

    auto alpha = 1.0f, beta = 0.0f;

    double exponentialAverageFactor = 1.0 /(n+1);



    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(handle,
        batchNormMode,
        &alpha,
        &beta,
        input_descriptor,
        input->data,
        input_descriptor,
        output->data,
        derivedBnDesc,
        bnScale,
        bnBias,
        exponentialAverageFactor,
        resultRunningMean,
        resultRunningVariance,
        0.00001,
        *mean_p,
        *Variance_p));


    free(bnScalec);
    free(bnBiasc);


    cudaFree(bnScale);
    cudaFree(bnBias);

    cudaFree(resultRunningMean);
    cudaFree(resultRunningVariance);

    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;



}

int DLGpuBatchNormalizationBackward(const DLArrayHandle input,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnBatchNormMode_t batchNormMode,
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace) {

    //assert(input->ndim==4||input->ndim==3||input->ndim==5);

    //handle
    //cudnnHandle_t handle;
    //CUDNN_CALL(cudnnCreate(&handle));
    cudnnHandle_t handle = (cudnnHandle_t)(*cudnnHandle);

    //input
    cudnnTensorDescriptor_t input_descriptor = (cudnnTensorDescriptor_t)((*cudnnlist)[0]);

    //filter
    cudnnTensorDescriptor_t bnScaleBiasDiffDesc = (cudnnTensorDescriptor_t)((*cudnnlist)[1]);

      size_t *s = (size_t *)((*cudnnlist)[2]);
    //��ķ�ʽ��������padding
    float *bnScalec = (float*)malloc(*s);
    for(int i=0;i< *s / sizeof(float);i++){
        bnScalec[i] = 1.0;
    }
    float *bnScale;
    float *resultBnScaleDiff;
    float *resultBnBiasDiff;

    int mycomputesapce=3*((int) *s);
    if((mycomputesapce>computespace)&&(computespace!=-1))
    {
       //printf("\n%d,%d",mycomputesapce,computespace);
       *memorytoSaving=mycomputesapce-computespace;
       return 0;
    }

    cudaError_t e = cudaMalloc((void**)&bnScale, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;

     
    return 0;
    }
    e = cudaMalloc((void**)&resultBnScaleDiff, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;
      cudaFree(bnScale);
   
    return 0;
    }
    e = cudaMalloc((void**)&resultBnBiasDiff, *s);

    if ((e != cudaSuccess) && (e != cudaErrorCudartUnloading)){
      //内存超了：
      *memorytoSaving = (int) *s;
      cudaFree(bnScale);
      cudaFree(resultBnScaleDiff);
    
    return 0;
    }



    cudaMemcpy(bnScale, bnScalec, *s, cudaMemcpyHostToDevice);

    auto alpha = 1.0f, beta = 0.0f;





    CUDNN_CALL(cudnnBatchNormalizationBackward(handle,
        batchNormMode,
        &alpha,
        &beta,
        &alpha,
        &beta,
        input_descriptor,
        input->data,
        input_descriptor,
        doutput->data,
        input_descriptor,
        dinput->data,
        bnScaleBiasDiffDesc,
        bnScale,
        resultBnScaleDiff,
        resultBnBiasDiff,
        0.00001,
        *mean_p,
        *Variance_p));
    free(bnScalec);
    cudaFree(bnScale);
    cudaFree(resultBnBiasDiff);
    cudaFree(resultBnScaleDiff);
    cudaFree(*mean_p);
    cudaFree(*Variance_p);
    // CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasDiffDesc));
    //CUDNN_CALL(cudnnDestroy(handle));
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;



}


__global__ void adam_kernel(float* outputArr, float* mArr, float* vArr, 
                            float b1t,float b2t,float e,float learning_rate,
                            int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
    outputArr[index] = outputArr[index] - learning_rate * mArr[index] / (1 - b1t) / (sqrt(vArr[index] / (1 - b2t)) + e);
    }
}



int DLGpuAdam(DLArrayHandle output,
              const DLArrayHandle m, const DLArrayHandle v,
              float b1t,float b2t,float e,float learning_rate, void **cudaStream) {
    assert(m->ndim == output->ndim);
    assert(v->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < output->ndim; ++i) {
        count *= output->shape[i];
    }
    float* mData = (float*) m->data;
    float* vData = (float*) v->data;
    float* outputData = (float*) output->data;
    adam_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
        outputData, mData, vData,  
        b1t, b2t, e, learning_rate,
        count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}





__global__ void adam_mv_kernel(float* mArr, float* vArr, float* gArr,
                                float b1,float b2, int count, int reduce) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        if(index<reduce){
            mArr[index] = mArr[index] * b1 +  gArr[index] * (1 - b1);
        }
        else{
            int realindex = index-reduce;
            vArr[realindex] = vArr[realindex] * b2 +  pow(gArr[realindex],2) * (1 - b2);
        }
            
    }
}



int DLGpuAdam_mv(DLArrayHandle m,
                DLArrayHandle v,
                const DLArrayHandle g,
                float b1,
                float b2, void **cudaStream) {
    assert(v->ndim == g->ndim);
    assert(m->ndim == g->ndim);
    int count = 1;
    for (int i = 0; i < g->ndim; ++i) {
        count *= g->shape[i];
    }

    float* mData = (float*) m->data;
    float* vData = (float*) v->data;
    float* gData = (float*) g->data;
    adam_mv_kernel<<<BLOCK_NUM(count*2), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    mData, vData, gData,
    b1,b2,
    count*2,
    count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}




__global__ void cross_kernel(float* xArr, float* yArr, float* outputArr,
                                    float b, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
    outputArr[index] = log(xArr[index])  * yArr[index] * (-1) * b;
    }
}


int DLGpuCross(const DLArrayHandle x,
                    const DLArrayHandle y,
                    DLArrayHandle output,
                    float b, void **cudaStream) {
    assert(x->ndim == output->ndim);
    assert(x->ndim == y->ndim);
    int count = 1;
    for (int i = 0; i < output->ndim; ++i) {
        count *= output->shape[i];
    }
    float* xData = (float*) x->data;
    float* yData = (float*) y->data;
    float* outputData = (float*) output->data;
    cross_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    xData, yData, outputData,
    b,
    count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}


__global__ void cross_backward_kernel(float* xArr, float* yArr, float* doutputArr, float* outputArr,
                                    float b, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
    outputArr[index] =  yArr[index] * (-1) * b * (1.0 / xArr[index]) * doutputArr[index];
    }
}


int DLGpuCrossBackward(const DLArrayHandle x,
                    const DLArrayHandle y,
                    const DLArrayHandle doutput,
                    DLArrayHandle output,
                    float b, void **cudaStream) {
    assert(x->ndim == output->ndim);
    assert(x->ndim == y->ndim);
    int count = 1;
    for (int i = 0; i < output->ndim; ++i) {
        count *= output->shape[i];
    }
    float* xData = (float*) x->data;
    float* yData = (float*) y->data;
    float* doutputData = (float*) doutput->data;
    float* outputData = (float*) output->data;
    cross_backward_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    xData, yData, doutputData, outputData,
    b,
    count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}


__global__ void sgd_update_kernel(float* outputArr, float* mArr, 
                                    float b, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
    outputArr[index] = outputArr[index]  - b * mArr[index] ;
    }
}



int DLGpuSgdUpdate(DLArrayHandle output,
                    const DLArrayHandle m,
                    float b, void **cudaStream) {
    assert(m->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < output->ndim; ++i) {
        count *= output->shape[i];
    }
    float* mData = (float*) m->data;
    float* outputData = (float*) output->data;
    sgd_update_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM, 0, (cudaStream_t)*cudaStream>>>(
    outputData, mData, 
    b,
    count);
    cudaStreamSynchronize((cudaStream_t)*cudaStream);
    return 0;
}


int DLGpuGetIndextoVaribaleNumberCudaPointer(int *index_to_number,
                                            int *prefix,
                                            int count,
                                            void *** result) {

    int *cuda_index_to_number, *cuda_prefix;

    cudaMalloc((void**)&cuda_index_to_number, sizeof(int)*count);
    cudaMalloc((void**)&cuda_prefix, sizeof(int)*count);

    cudaMemcpy(cuda_index_to_number,index_to_number,sizeof(int)*count,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_prefix,prefix,sizeof(int)*count,cudaMemcpyHostToDevice);

    *result = (void **)malloc(sizeof(int*)*2);

    (*result)[0] = cuda_index_to_number;
    (*result)[1] = cuda_prefix;


    cudaDeviceSynchronize();
    return 0;

}


int DLGpuGetN2CudaPointer(void** output, void** g, int number,void **** result){


    float** outputData,** gData;
    cudaMalloc((void **)&outputData,sizeof(float*)*(number));
    cudaMalloc((void **)&gData,sizeof(float*)*(number));

    cudaMemcpy(outputData,output,sizeof(float*)*(number),cudaMemcpyHostToDevice);
    cudaMemcpy(gData,g,sizeof(float*)*(number),cudaMemcpyHostToDevice);

    *result = (void ***)malloc(sizeof(float**)*2);

    (*result)[0] = (void **)outputData;
    (*result)[1] = (void **)gData;


    cudaDeviceSynchronize();
    return 0;

}

int DLGpuGetN4CudaPointer(void** output, void** m, void** v, void** g, int number,void **** result){


    float** outputData,** mData,** vData,** gData;
    cudaMalloc((void **)&outputData,sizeof(float*)*(number));
    cudaMalloc((void **)&mData,sizeof(float*)*(number));
    cudaMalloc((void **)&vData,sizeof(float*)*(number));
    cudaMalloc((void **)&gData,sizeof(float*)*(number));

    cudaMemcpy(outputData,output,sizeof(float*)*(number),cudaMemcpyHostToDevice);
    cudaMemcpy(mData,m,sizeof(float*)*(number),cudaMemcpyHostToDevice);
    cudaMemcpy(vData,v,sizeof(float*)*(number),cudaMemcpyHostToDevice);
    cudaMemcpy(gData,g,sizeof(float*)*(number),cudaMemcpyHostToDevice);

    *result = (void***)malloc(sizeof(float**)*4);

    (*result)[0] = (void **)outputData;
    (*result)[1] = (void **)mData;
    (*result)[2] = (void **)vData;
    (*result)[3] = (void **)gData;



    cudaDeviceSynchronize();
    return 0;

}


// __global__ void adam_mv_o_1_kernel(float** mArr, float** vArr, float** gArr, const int *index_to_number, const int *prefix,
//                                 const float b1, const float b2, const int count, const int reduce) {
//     CUDA_1D_KERNEL_LOOP(index, count) {
//         if(index<reduce){
//             int realnumber = index_to_number[index];
//             int realArrindex = prefix[index];
//             (mArr[realnumber])[realArrindex] = (mArr[realnumber])[realArrindex] * b1 +  (gArr[realnumber])[realArrindex] * (1 - b1);
//         }
//         else{
//             int realindex = index-reduce;
//             int realnumber = index_to_number[realindex];
//             int realArrindex = prefix[realindex];
//             (vArr[realnumber])[realArrindex] = (vArr[realnumber])[realArrindex] * b2 +  pow((gArr[realnumber])[realArrindex],2) * (1 - b2);
//         }

//     }
// }

// __global__ void adam_mv_o_2_kernel(float** outputArr, float** mArr, float** vArr, const int *index_to_number, const int *prefix,
//                                     const float b1t, const float b2t, const float e, const float learning_rate, const int count) {
//     CUDA_1D_KERNEL_LOOP(index, count) {

//         int realnumber = index_to_number[index];
//         int realArrindex = prefix[index];
//         (outputArr[realnumber])[realArrindex] = ((outputArr[realnumber])[realArrindex]) -
//          learning_rate * (mArr[realnumber])[realArrindex] / (1 - b1t) / (sqrt((vArr[realnumber])[realArrindex] / (1 - b2t)) + e);



//     }
// }











__global__ void adam_mv_o_1_kernel(float** outputArr, float** mArr, float** vArr, float **gArr, const int *index_to_number, const int *prefix,
    const float b1, const float b2, const float b1t, const float b2t, const float e, const float learning_rate, const int count) {

    __shared__ float  mv[MAX_THREADS_NUM];


    CUDA_1D_KERNEL_LOOP(index, count) {
        int realindex = index/2;
        int ism = index%2;
        int realnumber,realArrindex;
        if(ism==0){
            realnumber = index_to_number[realindex];
            realArrindex = prefix[realindex];
            float mm = (mArr[realnumber])[realArrindex] * b1 +  (gArr[realnumber])[realArrindex] * (1 - b1);
            (mArr[realnumber])[realArrindex] = mm;
            mv[threadIdx.x] = learning_rate * mm / (1 - b1t);


        }
        else{
            realnumber = index_to_number[realindex];
            realArrindex = prefix[realindex];
            float vv = (vArr[realnumber])[realArrindex] * b2 +  pow((gArr[realnumber])[realArrindex],2) * (1 - b2);
            (vArr[realnumber])[realArrindex] = vv;
            mv[threadIdx.x] = sqrt(vv / (1 - b2t)) + e;
        }

        __syncthreads();

        if(ism==0){

            (outputArr[realnumber])[realArrindex] -= mv[threadIdx.x] / mv[threadIdx.x+1];

        }

    }
}





// int DLGpuAdam_o(float *** n4list,
//                 const int **indexlist,
//                 const int number,
//                 const int count,
//                 const float b1,
//                 const float b2,
//                 const float b1t,
//                 const float b2t,
//                 const float e,
//                 const float learning_rate) {

//     // clock_t start, finish;
//     // start = clock();



//     adam_mv_o_1_kernel<<<BLOCK_NUM(count*2), MAX_THREADS_NUM>>>(
//         n4list[1], n4list[2], n4list[3], indexlist[0], indexlist[1],
//         b1,b2,
//         count*2,
//         count);

//     adam_mv_o_2_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
//         n4list[0], n4list[1], n4list[2], indexlist[0], indexlist[1],
//         b1t,b2t,e,learning_rate,
//         count);


//     // finish = clock();
//     // double duration = (double)(finish - start) / CLOCKS_PER_SEC;
//     // printf( "Total elapsed %lf seconds\n", duration);
//     cudaDeviceSynchronize();
  //  return 0;

// }


__global__ void sgd_o_1_kernel(float** outputArr, float** gArr, const int *index_to_number, const int *prefix,
                                 const float learning_rate, const int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {

        int realnumber = index_to_number[index];
        int realArrindex = prefix[index];
        (outputArr[realnumber])[realArrindex] = ((outputArr[realnumber])[realArrindex]) -
         learning_rate * (gArr[realnumber])[realArrindex];



    }
}



int DLGpuSgd_o(void **** n2list,
                const void ***indexlist,
                const int count,
                const float learning_rate) {

    // clock_t start, finish;
    // start = clock();



    sgd_o_1_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    (float**)((*n2list)[0]), (float**)((*n2list)[1]), (int *)((*indexlist)[0]), (int *)((*indexlist)[1]),
    learning_rate,
    count);



    // finish = clock();
    // double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    // printf( "Total elapsed %lf seconds\n", duration);
    cudaDeviceSynchronize();
    return 0;

}

int DLGpuAdam_o(void **** n4list,
                const void ***indexlist,
                const int count,
                const float b1,
                const float b2,
                const float b1t,
                const float b2t,
                const float e,
                const float learning_rate) {

    // clock_t start, finish;
    // start = clock();



    adam_mv_o_1_kernel<<<BLOCK_NUM(count*2), MAX_THREADS_NUM>>>(
        (float**)((*n4list)[0]), (float**)((*n4list)[1]), (float**)((*n4list)[2]),(float**)((*n4list)[3]), (int *)((*indexlist)[0]), (int *)((*indexlist)[1]),
        b1,b2,b1t,b2t,e,learning_rate,
        count*2);



    // finish = clock();
    // double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    // printf( "Total elapsed %lf seconds\n", duration);
    cudaDeviceSynchronize();
    return 0;

}


int getInt(int *intp){
    return *intp;
}



#include <assert.h>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(float        *h_a,
                   float        *h_b,
                   float        *d,
                   unsigned int  n,
                   char         *desc,
                   int        *in,
                   int        *out)
{
  //printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  //printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
  *in=(int)(bytes / time);
  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  //printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
  *out=(int)(bytes / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int testPcie(int *in,int *out)
{
  unsigned int nElements = 4*1024*1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);                    // host pageable
  h_bPageable = (float*)malloc(bytes);                    // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  //printf("\nDevice: %s\n", prop.name);
  //printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable",in,out);
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned",in,out);

 // printf("\n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}






