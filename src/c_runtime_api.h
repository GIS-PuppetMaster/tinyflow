/*!
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef TINYFLOW_RUNTIME_C_RUNTIME_API_H_
#define TINYFLOW_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define TINYFLOW_EXTERN_C extern "C"
#else
#define TINYFLOW_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>
TINYFLOW_EXTERN_C {
  /*! \brief type of array index. */
  typedef int64_t index_t;

  /*! \brief the array handle */
  typedef DLArray *DLArrayHandle;
  /*!
   * \brief The stream that is specific to device
   * can be NULL, which indicates the default one.
   */
  typedef void *DLStreamHandle;

  typedef enum {
      SAME = 0,
      VALID = 1,
  } paddingStatus_t;

//  typedef enum {
//      CUDNN_TENSOR_NCHW = 0,
//      CUDNN_TENSOR_NHWC = 1,
//  } cudnnTensorFormat_t

  // Array related apis for quick proptying
  /*!
   * \brief Allocate a nd-array's memory,
   *  including space of shape, of given spec.
   *
   * \param shape The shape of the array, the data content will be copied to out
   * \param ndim The number of dimension of the array.
   * \param ctx The ctx this array sits on.
   * \param out The output handle.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                   DLArrayHandle *out, int *memorytoSaving);

  /*!
   * \brief Free the DL Array.
   * \param handle The array handle to be freed.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayFree(DLArrayHandle handle);

  /*!
   * \brief Copy the array, both from and to must be valid during the copy.
   * \param from The array to be copied from.
   * \param to The target space.
   * \param stream The stream where the copy happens, can be NULL.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                        DLStreamHandle stream);

  /*!
   * \brief Set all array elements to given value.
   * \param arr The array to be Set.
   * \param value The target value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuArraySet(DLArrayHandle arr, float value, void **cudaStream);

  /*!
   * \brief Broadcast input array to output array.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuBroadcastTo0(const DLArrayHandle input, DLArrayHandle output, void **cudaStream);

  int DLGpuBroadcastToBackward0(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuBroadcastTo1(const DLArrayHandle input, DLArrayHandle output, void **cudaStream);

  int DLGpuBroadcastToBackward1(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Reduce sum input array by axis=0 and store to output.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Elementwise add two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output, void **cudaStream);

  /*!
   * \brief Add matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                       DLArrayHandle output, void **cudaStream);

  /*!
   * \brief Elementwise multiply two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseMultiply(
      const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output, void **cudaStream);

  /*!
   * \brief Multiply matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                                 DLArrayHandle output, void **cudaStream);

  /*!
   * \brief Matrix multiply two matrices and store to output.
   * \param matA The left input array.
   * \param transposeA Whether matA needs to be transposed
   * \param matB The right input array.
   * \param transposeB Whether matB needs to be transposed
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                          const DLArrayHandle matB, bool transposeB,
                          DLArrayHandle matC, void **cublasHandle, void **cudaStream);

  /*!
   * \brief Compute relu on all array elements, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Compute relu gradient, and store to output.
   * \param input The input array.
   * \param in_grad The input gradients value.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                        DLArrayHandle output);

  /*!
   * \brief Compute softmax on matrix, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output, void **cudaStream);

  /*!
   * \brief Compute softmax_cross_entropy.
   *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
   * \param input_a The y array.
   * \param input_b The y_ array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output, void **cudaStream);




  int DLGpuMatrixExp(const DLArrayHandle input, DLArrayHandle output);


  int DLGpuMatrixLog(const DLArrayHandle input, DLArrayHandle output, void **cudaStream);


  int DLGpuMatrixReverse(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuMatrixPow(const DLArrayHandle input,const float val, DLArrayHandle output);



  int DLGpuReduceSumAxisN(const DLArrayHandle input, DLArrayHandle output, const int axis);


  int DLGpuReduceSumAll(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuReduceSumAllBackward(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuReduceSumAxisNBackward(const DLArrayHandle input, DLArrayHandle output, const int axis);

  int DLGReduceSumGetCudnnlist(const int *input_shapes,
      const int *output_shapes,
      const int sizeofshape,
      cudnnTensorFormat_t dataformat,
      void *** cudnnlist,
      void **cudnnHandle);
  int DLGpuReduceSum(const DLArrayHandle input, DLArrayHandle output, void ***cudnnlist, void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);

  int DLGpuConcatForward(const DLArrayHandle input1,const DLArrayHandle input2,  DLArrayHandle output, void **cudaStream);
  int DLGpuConcataBackward(const DLArrayHandle input1,const DLArrayHandle input2,const DLArrayHandle doutput,DLArrayHandle dinput1, void **cudaStream);
  int DLGpuConcatbBackward(const DLArrayHandle input1,const DLArrayHandle input2,const DLArrayHandle doutput,DLArrayHandle dinput2, void **cudaStream);

  int DLGpuCreatecudaStream(void **cudaStream);
  int DLGpuDestroycudaStream(void **cudaStream);
  int DLGpuCreatecudnnHandle(void **cudnnHandle, void **cudaStream);
  int DLGpuDestroycudnnHandle(void **cudnnHandle);
  int DLGpuCreatecublasHandle(void **cublasHandle, void **cudaStream);
  int DLGpuDestroycublasHandle(void **cublasHandle);
//juanji
  int DLGpuConvolution1DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      void ***cudnnlist,
      void **cudnnHandle, int *memorytoSaving);


  int DLGpuConvolutionBackwardFilter(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      void*** cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);

  int DLGpuConvolutionBackwardData(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dinput,
      void*** cudnnlist,
      void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);



  int DLGpuConvolution2DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      void ***cudnnlist,/* horizontal filter stride */
      void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);


  int DLGpuConvolution3DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving);

   int DLGpuConvolution1DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      void ***cudnnlist,
      void **cudnnHandle);



  int DLGpuConvolution2DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      void ***cudnnlist,/* horizontal filter stride */
      void **cudnnHandle);


  int DLGpuConvolution3DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      void ***cudnnlist,
      void **cudnnHandle);


  int DLGpuPooling1DForward(const DLArrayHandle input,
      DLArrayHandle output,
      void ***cudnnlist,
      void **cudnnHandle);

  int DLGpuPooling2DForward(const DLArrayHandle input,
      DLArrayHandle output,
      void ***cudnnlist,
      void **cudnnHandle, void **cudaStream);


  int DLGpuPooling3DForward(const DLArrayHandle input,
      DLArrayHandle output,
      void ***cudnnlist,
      void **cudnnHandle);



  int DLGpuPooling1DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
     void ***cudnnlist,
     void **cudnnHandle);

  int DLGpuPooling2DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream);


  int DLGpuPooling3DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void ***cudnnlist,
    void **cudnnHandle);


  int DLGpuConvolution1DForwardGetOutShape(const int* input_shapes,
      const int* filter_shapes,
      int* output_shapes,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int v,
      void ***cudnnlist);

  int DLGpuActivationForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream);

  int DLGpuActivationBackward(const DLArrayHandle input,
    DLArrayHandle dinput,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream);


  int DLGpuGetInputDescriptor(const int *input_shapes,
    const int sizeofshape,
    cudnnTensorFormat_t dataformat,
    void ***inputd);

  int DLGpuActivationGetCudnnlist(const int *input_shapes,
    const int sizeofshape,
    cudnnTensorFormat_t dataformat,
    cudnnActivationMode_t activationMode,
    void ***cudnnlist);

  int DLGpuPooling1DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding_w,
    const int v,
    const int filter_w,
    void ***cudnnlist);

  int DLGpuConvolution2DForwardGetOutShape(const int* input_shapes,
      const int* filter_shapes,
      int* output_shapes,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int u,          /* vertical filter stride */
      const int v,
      void ***cudnnlist);

  // int Test(cudnnTensorDescriptor_t i);
  

  int DLGpuConvolution3DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int s1,
    const int s2,
    const int s3,
    void ***cudnnlist);

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
    void ***cudnnlist);

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
    void ***cudnnlist);


  int DLGpuDropoutForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const float dropout,
    const int seed,
    void **reserveSpace_p,
    void ***inputd,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);

  int DLGpuDropoutBackward(const DLArrayHandle doutput,
    DLArrayHandle dinput,
    void **reserveSpace_p,
    void ***cudnnlist,
    void **cudnnHandle, void **cudaStream);


  int DLGpuCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1loss(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL2loss(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1LossGradient(const DLArrayHandle input,
                               const DLArrayHandle input1,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuL2LossGradient(const DLArrayHandle input,
                               const DLArrayHandle input1,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1regular(const DLArrayHandle input_a,
                               DLArrayHandle output);
  int DLGpuL2regular(const DLArrayHandle input_a,
                               DLArrayHandle output);
  int DLGpuL1regularGradient(const DLArrayHandle input,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuL2regularGradient(const DLArrayHandle input,
                               const DLArrayHandle grad,
                               DLArrayHandle output);



  int DLGpuBatchNormalizationGetCudnnlist(const int *input_shapes,
    int sizeofshape,
    cudnnTensorFormat_t dataformat,
    cudnnBatchNormMode_t batchNormMode,
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist);

  int DLGpuBatchNormalizationForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnBatchNormMode_t batchNormMode,
    int n,//第n+1次使用
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);


  int DLGpuBatchNormalizationBackward(const DLArrayHandle input,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnBatchNormMode_t batchNormMode,
    void **mean_p,
    void **Variance_p,
    void ***cudnnlist,
    void **cudnnHandle, int *memorytoSaving, void **cudaStream,int computespace);
  
  int DLGpuAdam(DLArrayHandle output,
                const DLArrayHandle m, const DLArrayHandle v,
                float b1t,float b2t,float e,float learning_rate, void **cudaStream);
  


  int DLGpuAdam_mv(DLArrayHandle m,
                DLArrayHandle v,
                const DLArrayHandle g,
                float b1,
                float b2, void **cudaStream);

  int DLGpuAdam_o(void **** n4list,
                const void ***indexlist,
                const int count,
                const float b1,
                const float b2,
                const float b1t,
                const float b2t,
                const float e,
                const float learning_rate);

  int DLGpuSgd_o(void **** n2list,
                const void ***indexlist,
                const int count,
                const float learning_rate);
  
  int DLGpuSgdUpdate(DLArrayHandle output,
                    const DLArrayHandle m,
                   // const int* shape_prefix,
                    //const int number,
                    float b, void **cudaStream);
  int DLGpuCross(const DLArrayHandle x,
                    const DLArrayHandle y,
                    DLArrayHandle output,
                    float b, void **cudaStream);
  int DLGpuCrossBackward(const DLArrayHandle x,
                    const DLArrayHandle y,
                    const DLArrayHandle doutput,
                    DLArrayHandle output,
                    float b, void **cudaStream);


  int DLGpuGetIndextoVaribaleNumberCudaPointer(int *index_to_number,
                                            int *prefix,
                                            int count,
                                            void *** result);

  int DLGpuGetN4CudaPointer(void** output, void** m, void** v, void** g, int number,void **** result);

  int DLGpuGetN2CudaPointer(void** output, void** g, int number,void **** result);

  int getInt(int *intp);
  int testPcie(int *in,int *out);
  int SetCudaMemoryLimit(size_t size);
//  int DLGpuSgdUpdate(DLArrayHandle* output,
//                    const DLArrayHandle* m,
//                    const int* shape_prefix,
//                    const int number,
//                    float b);

} // TINYFLOW_EXTERN_C

#endif // TINYFLOW_RUNTIME_C_RUNTIME_API_H_
