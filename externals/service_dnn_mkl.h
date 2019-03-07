/* file: service_dnn_mkl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Template wrappers for common Intel(R) MKL functions.
//--
*/


#ifndef __SERVICE_DNN_MKL_H__
#define __SERVICE_DNN_MKL_H__

#include "daal_defines.h"
#include "mkl_dnn_types.h"
#include "mkl_daal.h"

#include "service_dnn.h"

#if !defined(__DAAL_CONCAT4)
    #define __DAAL_CONCAT4(a,b,c,d) __DAAL_CONCAT41(a,b,c,d)
    #define __DAAL_CONCAT41(a,b,c,d) a##b##c##d
#endif

#if defined(__APPLE__)
    #define __DAAL_MKL_SSE2  ssse3_
    #define __DAAL_MKL_SSSE3 ssse3_
#else
    #define __DAAL_MKL_SSE2  sse2_
    #define __DAAL_MKL_SSSE3 ssse3_
#endif

#define __DAAL_DNNFN(f_cpu,f_pref,f_name)        __DAAL_CONCAT4(fpk_,f_pref,f_cpu,f_name)
#define __DAAL_DNNFN_CALL(f_pref,f_name,f_args)  __DAAL_DNNFN_CALL1(f_pref,f_name,f_args)

#if (defined(__x86_64__) && !defined(__APPLE__)) || defined(_WIN64)
    #define __DAAL_MKLFPK_KNL   avx512_mic_
#else
    #define __DAAL_MKLFPK_KNL   avx2_
#endif


#define __DAAL_DNNFN_CALL1(f_pref,f_name,f_args)                    \
    if(avx512 == cpu)                                               \
    {                                                               \
        return __DAAL_DNNFN(avx512_,f_pref,f_name) f_args;          \
    }                                                               \
    if(avx512_mic == cpu)                                           \
    {                                                               \
        return __DAAL_DNNFN(__DAAL_MKLFPK_KNL,f_pref,f_name) f_args; \
    }                                                               \
    if(avx2 == cpu)                                                 \
    {                                                               \
        return __DAAL_DNNFN(avx2_,f_pref,f_name) f_args;            \
    }                                                               \
    if(avx == cpu)                                                  \
    {                                                               \
        return __DAAL_DNNFN(avx_,f_pref,f_name) f_args;             \
    }                                                               \
    if(sse42 == cpu)                                                \
    {                                                               \
        return __DAAL_DNNFN(sse42_,f_pref,f_name) f_args;           \
    }                                                               \
    if(ssse3 == cpu)                                                \
    {                                                               \
        return __DAAL_DNNFN(__DAAL_MKL_SSSE3,f_pref,f_name) f_args; \
    }                                                               \
    if(true)  /* default; sse2 == cpu */                            \
    {                                                               \
        return __DAAL_DNNFN(__DAAL_MKL_SSE2,f_pref,f_name) f_args;  \
    }

namespace daal
{
namespace internal
{
namespace mkl
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu>
struct MklDnn{};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklDnn<double, cpu>
{
    typedef dnnError_t     ErrorType;
    typedef dnnPrimitive_t PrimitiveType;
    typedef dnnLayout_t    LayoutType;
    typedef dnnAlgorithm_t AlgorithmType;
    typedef dnnBorder_t    BorderType;
    typedef size_t         SizeType;

    static dnnError_t xConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateForwardBias_F64, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                   convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardData_F64, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                    convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardFilter_F64, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                      convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t dstSize[])
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardBias_F64, (pConvolution, NULL, algorithm, nGroups, dimension, dstSize));
    }

    static dnnError_t xSplitCreate(
        PrimitiveType *pSplit, size_t nDstTensors, const LayoutType dataLayout, size_t dstChannelSize[])
    {
        __DAAL_DNNFN_CALL(dnn_, SplitCreate_F64, (pSplit, NULL, nDstTensors, dataLayout, dstChannelSize));
    }

    static dnnError_t xConcatCreate(
        PrimitiveType *pConcat, const size_t nSrcTensors, LayoutType src[])
    {
        __DAAL_DNNFN_CALL(dnn_, ConcatCreate_F64, (pConcat, NULL, nSrcTensors, src));
    }

    static dnnError_t xReLUCreateForward(
        dnnPrimitive_t *pRelu, const dnnLayout_t dataLayout, double negativeSlope)
    {
        __DAAL_DNNFN_CALL(dnn_, ReLUCreateForward_F64, (pRelu, NULL, dataLayout, negativeSlope));
    }

    static dnnError_t xReLUCreateBackward(
        dnnPrimitive_t *pRelu, const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, double negativeSlope)
    {
        __DAAL_DNNFN_CALL(dnn_, ReLUCreateBackward_F64, (pRelu, NULL, diffLayout, dataLayout, negativeSlope));
    }

    static ErrorType xLRNCreateForward(
        PrimitiveType *pLrn, const LayoutType dataLayout, size_t kernelSize, double alpha, double beta, double k)
    {
        __DAAL_DNNFN_CALL(dnn_, LRNCreateForward_F64, (pLrn, NULL, dataLayout, kernelSize, alpha, beta, k));
    }

    static ErrorType xLRNCreateBackward(
        PrimitiveType *pLrn, const LayoutType diffLayout, const LayoutType dataLayout, size_t kernelSize, double alpha, double beta, double k)
    {
        __DAAL_DNNFN_CALL(dnn_, LRNCreateBackward_F64, (pLrn, NULL, diffLayout, dataLayout, kernelSize, alpha, beta, k));
    }

    static ErrorType xPoolingCreateForward(
        PrimitiveType *pPooling, AlgorithmType op, const LayoutType srcLayout, const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const BorderType border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, PoolingCreateForward_F64, (pPooling, NULL, op, srcLayout, kernelSize, kernelStride, inputOffset, border_type));
    }

    static ErrorType xPoolingCreateBackward(
        PrimitiveType *pPooling, AlgorithmType op, const dnnLayout_t srcLayout, const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const BorderType border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, PoolingCreateBackward_F64, (pPooling, NULL, op, srcLayout, kernelSize, kernelStride, inputOffset, border_type));
    }

    static dnnError_t xExecute(dnnPrimitive_t primitive, void *resources[])
    {
        __DAAL_DNNFN_CALL(dnn_, Execute_F64, (primitive, resources));
    }
    static dnnError_t xConversionExecute(dnnPrimitive_t conversion, void *from, void *to)
    {
        if(!conversion) return E_SUCCESS;
        __DAAL_DNNFN_CALL(dnn_, ConversionExecute_F64, (conversion, from, to));
    }
    static dnnError_t xLayoutCreate(dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[])
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCreate_F64, (pLayout, dimension, size, strides));
    }
    static dnnError_t xLayoutCreateFromPrimitive(dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCreateFromPrimitive_F64, (pLayout, primitive, type));
    }
    static dnnError_t xAllocateBuffer(void **pPtr, dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, AllocateBuffer_F64, (pPtr, layout));
    }
    static dnnError_t xReleaseBuffer(void *ptr)
    {
        __DAAL_DNNFN_CALL(dnn_, ReleaseBuffer_F64, (ptr));
    }
    static int xLayoutCompare(const dnnLayout_t l1, const dnnLayout_t l2)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCompare_F64, (l1, l2));
    }
    static dnnError_t xConversionCreate(dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
    {
        __DAAL_DNNFN_CALL(dnn_, ConversionCreate_F64, (pConversion, from, to));
    }
    static dnnError_t xDelete(dnnPrimitive_t primitive)
    {
        __DAAL_DNNFN_CALL(dnn_, Delete_F64, (primitive));
    }
    static dnnError_t xLayoutDelete(dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutDelete_F64, (layout));
    }
    static size_t xLayoutGetMemorySize(const dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutGetMemorySize_F64, (layout));
    }
};

/*
// Single precision functions definition
*/

template<CpuType cpu>
struct MklDnn<float, cpu>
{
    typedef dnnError_t     ErrorType;
    typedef dnnPrimitive_t PrimitiveType;
    typedef dnnLayout_t    LayoutType;
    typedef dnnAlgorithm_t AlgorithmType;
    typedef dnnBorder_t    BorderType;
    typedef size_t         SizeType;

    static dnnError_t xConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateForwardBias_F32, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                   convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardData_F32, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                    convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[],
        const dnnBorder_t border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardFilter_F32, (pConvolution, NULL, algorithm, nGroups, dimension, srcSize, dstSize, filterSize,
                                                                      convolutionStrides, inputOffset, border_type));
    }
    static dnnError_t xConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,
        const size_t dstSize[])
    {
        __DAAL_DNNFN_CALL(dnn_, GroupsConvolutionCreateBackwardBias_F32, (pConvolution, NULL, algorithm, nGroups, dimension, dstSize));
    }

    static dnnError_t xSplitCreate(
        PrimitiveType *pSplit, size_t nDstTensors, const LayoutType dataLayout, size_t dstChannelSize[])
    {
        __DAAL_DNNFN_CALL(dnn_, SplitCreate_F32, (pSplit, NULL, nDstTensors, dataLayout, dstChannelSize));
    }

    static dnnError_t xConcatCreate(
        PrimitiveType *pConcat, const size_t nSrcTensors, LayoutType src[])
    {
        __DAAL_DNNFN_CALL(dnn_, ConcatCreate_F32, (pConcat, NULL, nSrcTensors, src));
    }

    static dnnError_t xReLUCreateForward(
        dnnPrimitive_t *pRelu, const dnnLayout_t dataLayout, float negativeSlope)
    {
        __DAAL_DNNFN_CALL(dnn_, ReLUCreateForward_F32, (pRelu, NULL, dataLayout, negativeSlope));
    }

    static dnnError_t xReLUCreateBackward(
        dnnPrimitive_t *pRelu, const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope)
    {
        __DAAL_DNNFN_CALL(dnn_, ReLUCreateBackward_F32, (pRelu, NULL, diffLayout, dataLayout, negativeSlope));
    }

    static ErrorType xLRNCreateForward(
        PrimitiveType *pLrn, const LayoutType dataLayout, size_t kernelSize, float alpha, float beta, float k)
    {
        __DAAL_DNNFN_CALL(dnn_, LRNCreateForward_F32, (pLrn, NULL, dataLayout, kernelSize, alpha, beta, k));
    }

    static ErrorType xLRNCreateBackward(
        PrimitiveType *pLrn, const LayoutType diffLayout, const LayoutType dataLayout, size_t kernelSize, float alpha, float beta, float k)
    {
        __DAAL_DNNFN_CALL(dnn_, LRNCreateBackward_F32, (pLrn, NULL, diffLayout, dataLayout, kernelSize, alpha, beta, k));
    }

    static ErrorType xPoolingCreateForward(
        PrimitiveType *pPooling, AlgorithmType op, const LayoutType srcLayout, const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const BorderType border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, PoolingCreateForward_F32, (pPooling, NULL, op, srcLayout, kernelSize, kernelStride, inputOffset, border_type));
    }

    static ErrorType xPoolingCreateBackward(
        PrimitiveType *pPooling, AlgorithmType op, const dnnLayout_t srcLayout, const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const BorderType border_type)
    {
        __DAAL_DNNFN_CALL(dnn_, PoolingCreateBackward_F32, (pPooling, NULL, op, srcLayout, kernelSize, kernelStride, inputOffset, border_type));
    }

    static dnnError_t xExecute(dnnPrimitive_t primitive, void *resources[])
    {
        __DAAL_DNNFN_CALL(dnn_, Execute_F32, (primitive, resources));
    }
    static dnnError_t xConversionExecute(dnnPrimitive_t conversion, void *from, void *to)
    {
        if(!conversion) return E_SUCCESS;
        __DAAL_DNNFN_CALL(dnn_, ConversionExecute_F32, (conversion, from, to));
    }
    static dnnError_t xLayoutCreate(dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[])
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCreate_F32, (pLayout, dimension, size, strides));
    }
    static dnnError_t xLayoutCreateFromPrimitive(dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCreateFromPrimitive_F32, (pLayout, primitive, type));
    }
    static dnnError_t xAllocateBuffer(void **pPtr, dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, AllocateBuffer_F32, (pPtr, layout));
    }
    static dnnError_t xReleaseBuffer(void *ptr)
    {
        __DAAL_DNNFN_CALL(dnn_, ReleaseBuffer_F32, (ptr));
    }
    static int xLayoutCompare(const dnnLayout_t l1, const dnnLayout_t l2)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutCompare_F32, (l1, l2));
    }
    static dnnError_t xConversionCreate(dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
    {
        __DAAL_DNNFN_CALL(dnn_, ConversionCreate_F32, (pConversion, from, to));
    }
    static dnnError_t xDelete(dnnPrimitive_t primitive)
    {
        __DAAL_DNNFN_CALL(dnn_, Delete_F32, (primitive));
    }
    static dnnError_t xLayoutDelete(dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutDelete_F32, (layout));
    }
    static size_t xLayoutGetMemorySize(const dnnLayout_t layout)
    {
        __DAAL_DNNFN_CALL(dnn_, LayoutGetMemorySize_F32, (layout));
    }
};


} // namespace mkl
} // namespace internal
} // namespace daal

#endif
