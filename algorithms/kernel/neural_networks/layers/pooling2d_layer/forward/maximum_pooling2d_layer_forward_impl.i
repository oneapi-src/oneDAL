/* file: maximum_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward pooling layer
//--
*/

#ifndef __MAXIMUM_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __MAXIMUM_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "service_defines.h"
#include "threading.h"
#if defined (__INTEL_COMPILER)
  #include <immintrin.h>
#endif

#include "service_mkl_tensor.h"

using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace maximum_pooling2d
{
namespace forward
{
namespace internal
{

/* Default compute_max implementation */
template<typename T, CpuType cpu>
struct pooling2d_opt_t
{
    inline void compute_max( int n, int offset, const T* parg, T* pmax, int* pmaxidx )
    {
        for(int i = 0; i < n; i++)
        {
            if(parg[i] > *pmax)
            {
                *pmaxidx = offset + i;
                *pmax    = parg[i];
            }
        }
    }
};

#if defined (__INTEL_COMPILER)

    /* AVX512_ALL common specialization for all CPU from Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 and newer */
    #if __CPUID__(DAAL_CPU) >= __avx512_mic__
        #define AVX512_ALL DAAL_CPU
    #else
        #define AVX512_ALL avx512_mic
    #endif

    /* Emulation of blendv instruction on hardware before sse4.2 */
    #if __CPUID__(DAAL_CPU) < __sse42__
      # define __mm_blendv_ps( sA, sB, sMask ) _mm_or_ps( _mm_andnot_ps( sMask, sA ), _mm_and_ps( sB, sMask ))
      # define __mm_blendv_pd( dA, dB, dMask ) _mm_or_pd( _mm_andnot_pd( dMask, dA ), _mm_and_pd( dB, dMask ))
    #else
      # define __mm_blendv_ps( sA, sB, sMask ) _mm_blendv_ps( sA, sB, sMask )
      # define __mm_blendv_pd( dA, dB, dMask ) _mm_blendv_pd( dA, dB, dMask )
    #endif

    template<>
    struct pooling2d_opt_t<float, AVX512_ALL>
    {
        inline void compute_max( int n, int offset, const float* parg, float* pmax, int* pmaxidx )
        {
            int i = 0;

            /* For bigger than 16 loop size use unrolled vector implementation */
            /* where it is faster than default code */
            if(__builtin_expect(n>=16,0))
            {
                int n16 = n & ~(16-1);
                __m512 mmax = _mm512_set1_ps( *pmax );
                for( i = 0; i < n16; i+=16 )
                {
                    __m512    ma         = _mm512_loadu_ps( &parg[i] );
                    __m512    mamax      = _mm512_max_ps( ma, mmax );
                              mmax       = _mm512_set1_ps( _mm512_reduce_max_ps( mamax ) );
                    __mmask16 idx        = _mm512_cmp_ps_mask( ma, mmax, _CMP_EQ_UQ );
                              idx        = _mm_tzcnt_32(((unsigned int)idx) | 0xffff0000);
                              (*pmaxidx) = (idx<16)?(i+idx+offset):(*pmaxidx);
                }
                _mm512_mask_storeu_ps( pmax, 0x1, mmax );
            }
            for(; i < n; i++)
            {
                if(parg[i] > *pmax)
                {
                    *pmaxidx = offset + i;
                    *pmax    = parg[i];
                }
            }
        }
    };

    template<>
    struct pooling2d_opt_t<double, AVX512_ALL>
    {
        inline void compute_max( int n, int offset, const double* parg, double* pmax, int* pmaxidx )
        {
            int i = 0;

            /* For bigger than 8 loop size use unrolled vector implementation */
            /* where it is faster than default code */
            if(__builtin_expect(n>=8,0))
            {
                int n8 = n & ~(8-1);
                __m512d mmax = _mm512_set1_pd( *pmax );
                for( i = 0; i < n8; i+=8 )
                {
                    __m512d   ma         = _mm512_loadu_pd( &parg[i] );
                    __m512d   mamax      = _mm512_max_pd( ma, mmax );
                              mmax       = _mm512_set1_pd( _mm512_reduce_max_pd( mamax ) );
                    __mmask8  idx        = _mm512_cmp_pd_mask( ma, mmax, _CMP_EQ_UQ );
                              idx        = _mm_tzcnt_32(((unsigned int)idx) | 0xffffff00);
                              (*pmaxidx) = (idx<8)?(i+idx+offset):(*pmaxidx);
                }
                _mm512_mask_storeu_pd( pmax, 0x1, mmax );
            }
            for(; i < n; i++)
            {
                if(parg[i] > *pmax)
                {
                    *pmaxidx = offset + i;
                    *pmax    = parg[i];
                }
            }
        }
    };

    template<CpuType cpu>
    struct pooling2d_opt_t<float, cpu>
    {
        inline void compute_max( int n, int offset, const float* parg, float* pmax, int* pmaxidx )
        {
            /* For bigger than 12 loop size use default implementation */
            /* where it is faster than intrinsics */
            if(__builtin_expect(n>12,0))
            {
                for(int i = 0; i < n; i++)
                {
                    if(parg[i] > *pmax)
                    {
                        *pmaxidx = offset + i;
                        *pmax    = parg[i];
                    }
                }
            }
            else
            {
                const int one    = 1;
                const int newidx = offset;
                __m128i mone     = _mm_loadu_si32( &one );
                __m128i mnewidx  = _mm_loadu_si32( &newidx );
                __m128i mmaxidx  = _mm_loadu_si32( pmaxidx );
                __m128 mmax      = _mm_load_ss( pmax );

                for(int i = 0; i < n; i++)
                {
                    __m128 ma     = _mm_load_ss( &parg[i] );
                    __m128i mmask = _mm_castps_si128(_mm_cmpgt_ps( ma, mmax ));
                    mmaxidx       = _mm_castps_si128(__mm_blendv_ps( _mm_castsi128_ps(mmaxidx),
                                                                     _mm_castsi128_ps(mnewidx),
                                                                     _mm_castsi128_ps(mmask) ));
                    mmax          = _mm_max_ss( ma, mmax );
                    mnewidx       = _mm_add_epi32(mnewidx,mone);
                }

                _mm_store_ss( (float*)(pmaxidx), _mm_castsi128_ps(mmaxidx) );
                _mm_store_ss( pmax, mmax );
            }
        }
    };

    template<CpuType cpu>
    struct pooling2d_opt_t<double, cpu>
    {
        inline void compute_max( int n, int offset, const double* parg, double* pmax, int* pmaxidx )
        {
            /* For bigger than 8 loop size use default implementation */
            /* where it is faster than intrinsics */
            if(__builtin_expect(n>8,0))
            {
                for(int i = 0; i < n; i++)
                {
                    if(parg[i] > *pmax)
                    {
                        *pmaxidx = offset + i;
                        *pmax    = parg[i];
                    }
                }
            }
            else
            {
                const int one    = 1;
                const int newidx = offset;
                __m128i mone     = _mm_loadu_si32( &one );
                __m128i mnewidx  = _mm_loadu_si32( &newidx );
                __m128i mmaxidx  = _mm_loadu_si32( pmaxidx );
                __m128d mmax     = _mm_load_sd( pmax );

                for(int i = 0; i < n; i++)
                {
                    __m128d ma    = _mm_load_sd( &parg[i] );
                    __m128i mmask = _mm_castpd_si128(_mm_cmpgt_pd( ma, mmax ));
                    mmaxidx       = _mm_castpd_si128(__mm_blendv_pd( _mm_castsi128_pd(mmaxidx),
                                                                     _mm_castsi128_pd(mnewidx),
                                                                     _mm_castsi128_pd(mmask) ));
                    mmax          = _mm_max_sd( ma, mmax );
                    mnewidx       = _mm_add_epi32(mnewidx,mone);
                }

                _mm_store_ss( (float*)(pmaxidx), _mm_castsi128_ps(mmaxidx) );
                _mm_store_sd( pmax, mmax );
            }
        }
    };

#endif /* defined (__INTEL_COMPILER) */

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::initialize(const services::Collection<size_t>& inDims,
                                                             const services::Collection<size_t>& outDims)
{
    size_t dimension = inDims.size();

    outputSize    = new size_t[dimension];
    outputStrides = new size_t[dimension];

    outputSize   [0] = outDims[dimension-1];
    outputStrides[0] = 1;

    for(size_t i = 1; i < dimension; i++)
    {
        outputSize   [i] = outDims[dimension - 1 - i];
        outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
    }

    ltUserOutput = xDnnLayout(dimension, outputSize, outputStrides); ON_ERR(ltUserOutput.err);
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &dataTensor, Tensor &valueTensor,
        Tensor *selectedPosTensor, const Parameter &parameter)
{
    const Collection<size_t> &dims = dataTensor.getDimensions();
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    MklTensor<algorithmFPType> *dataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor*>(&dataTensor));
    MklTensor<algorithmFPType> *valueMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&valueTensor);
    MklTensor<double> *selectedPosMklTensorDouble = dynamic_cast<MklTensor<double>*>(selectedPosTensor);
    MklTensor<float> *selectedPosMklTensorFloat = dynamic_cast<MklTensor<float>*>(selectedPosTensor);

    if (dataMklTensor != NULL && (selectedPosMklTensorDouble != NULL || selectedPosMklTensorFloat != NULL))
    {
        algorithmFPType* maxPoolRes[dnnResourceNumber] = {0};

        dnnLayout_t inputLayout;
        dnnLayout_t workspaceLayout;
        dnnLayout_t resultLayout;

        inputLayout = (dnnLayout_t)dataMklTensor->getDnnLayout();
        maxPoolRes[dnnResourceSrc] = dataMklTensor->getDnnArray();

        dnnError_t err;

        if (maxPoolPrim == NULL)
        {
            const int inputOffset[2] = { (int)(-parameter.paddings.size[0]), (int)(-parameter.paddings.size[1]) };
            err = dnn::xPoolingCreateForward(&maxPoolPrim, dnnAlgorithmPoolingMax, inputLayout,
                                         parameter.kernelSizes.size, parameter.strides.size, inputOffset, dnnBorderZeros);
            ON_ERR(err);
        }

        err = dnn::xLayoutCreateFromPrimitive(&workspaceLayout, maxPoolPrim, dnnResourceWorkspace); ON_ERR(err);
        if (selectedPosMklTensorDouble != NULL)
        {
            selectedPosMklTensorDouble->setDnnLayout(workspaceLayout);
            maxPoolRes[dnnResourceWorkspace] = (algorithmFPType*)selectedPosMklTensorDouble->getDnnArray();
        }
        else
        {
            selectedPosMklTensorFloat->setDnnLayout(workspaceLayout);
            maxPoolRes[dnnResourceWorkspace] = (algorithmFPType*)selectedPosMklTensorFloat->getDnnArray();
        }

        if (valueMklTensor != NULL)
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, maxPoolPrim, dnnResourceDst); ON_ERR(err);
            valueMklTensor->setDnnLayout(resultLayout);
            maxPoolRes[dnnResourceDst] = valueMklTensor->getDnnArray();

            err = dnn::xExecute(maxPoolPrim, (void**)maxPoolRes); ON_ERR(err);
        }
        else
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, maxPoolPrim, dnnResourceDst); ON_ERR(err);

            WriteOnlySubtensor<algorithmFPType, cpu> valueBlock(valueTensor, 0, 0, 0, valueDims[0]);
            algorithmFPType *valueArray = valueBlock.get();

            LayoutConvertor<algorithmFPType, cpu> cvFromInnerOutput(&maxPoolRes[dnnResourceDst], resultLayout, false, &valueArray, ltUserOutput.get(), true); ON_ERR(cvFromInnerOutput.err);

            err = dnn::xExecute(maxPoolPrim, (void**)maxPoolRes); ON_ERR(err);

            cvFromInnerOutput.convert(); ON_ERR(cvFromInnerOutput.err);

            dnn::xLayoutDelete(resultLayout);
        }
    }
    else
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor(const_cast<Tensor&>(dataTensor), 0, 0, 0, dims[0]);
        DAAL_CHECK_BLOCK_STATUS(dataSubtensor);
        const algorithmFPType *data = dataSubtensor.get();

        WriteOnlySubtensor<algorithmFPType, cpu, Tensor> valueSubtensor(valueTensor, 0, 0, 0, valueDims[0]);
        DAAL_CHECK_BLOCK_STATUS(valueSubtensor);
        algorithmFPType *value = valueSubtensor.get();

        int *selectedPos = nullptr;
        WriteOnlySubtensor<int, cpu, Tensor> selectedPosSubtensor;
        if(parameter.predictionStage == false)
        {
            selectedPosSubtensor.set(selectedPosTensor, 0, 0, 0, valueDims[0]);
            DAAL_CHECK_BLOCK_STATUS(selectedPosSubtensor);
            selectedPos = selectedPosSubtensor.get();
            size_t selectedPosSize = selectedPosTensor->getSize();
            daal::services::internal::service_memset<int, cpu>(selectedPos, 0, selectedPosSize);
        }


        pooling2d::internal::Parameter par(parameter.indices.size, parameter.paddings.size,
                                           parameter.strides.size, parameter.kernelSizes.size,
                                           dataTensor, dims, valueDims);

        size_t nDim = dims.size();

        if (selectedPos)
        {
            if (par.firstIndex == nDim - 2 && par.secondIndex == nDim - 1 && par.firstPadding == 0 && par.secondPadding == 0)
            {
                indicesLastZeroPaddingsCompute(par, data, value, selectedPos);
            }
            else if (par.firstIndex == 0 && par.secondIndex == 1 && par.firstPadding == 0 && par.secondPadding == 0)
            {
                indicesFirstZeroPaddingsCompute(par, data, value, selectedPos);
            }
            else
            {
                defaultCompute(par, data, value, selectedPos);
            }
        }
        else
        {
            if (par.firstIndex == nDim - 2 && par.secondIndex == nDim - 1 && par.firstPadding == 0 && par.secondPadding == 0)
            {
                indicesLastZeroPaddingsCompute(par, data, value);
            }
            else if (par.firstIndex == 0 && par.secondIndex == 1 && par.firstPadding == 0 && par.secondPadding == 0)
            {
                indicesFirstZeroPaddingsCompute(par, data, value);
            }
            else
            {
                defaultCompute(par, data, value);
            }
        }
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr, int *selectedPosPtr)
{
    const algorithmFPType zero = 0.0;
    algorithmFPType max = -(services::internal::MaxVal<algorithmFPType>::get());
    int maxIdx = -1;

    /*
     * Loops over the kernel
     */
    const DAAL_INT fUpper = f + par.firstKernelSize;

    for (DAAL_INT fi = f; fi < fUpper; fi++)
    {
        const DAAL_INT sUpper = s + par.secondKernelSize;

        PRAGMA_NOVECTOR
        for (DAAL_INT si = s; si < sUpper; si++)
        {
            const DAAL_INT dataIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));

            const bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));
            const algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

            int _idx = (fi - f) * par.secondKernelSize + (si - s);
            if (dataValue > max)
            {
                max = dataValue;
                maxIdx = (fi - f) * par.secondKernelSize + (si - s);
            }
        }
    }
    valuePtr[j] = max;
    selectedPosPtr[j] = maxIdx;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesLastZeroPaddingsCompute(
                const pooling2d::internal::Parameter &par,
                const algorithmFPType *data, algorithmFPType *value,
                int *selectedPos)
{
    const algorithmFPType minValue = -(services::internal::MaxVal<algorithmFPType>::get());

    threader_for(par.offsetBefore, par.offsetBefore, [&](size_t i)
    {
        size_t ifos = par.firstOutSize * i;
        size_t ifs  = par.firstSize * i;

        for ( size_t fo = 0; fo < par.firstOutSize; fo++ )
        {
            size_t valueIndex          = par.secondOutSize * ( fo + ifos );
            algorithmFPType *valueBase = value + valueIndex;
            int *posBase               = selectedPos + valueIndex;

            for (size_t so = 0; so < par.secondOutSize; so++)
            {
                posBase[so]   = -1;
                valueBase[so] = minValue;
            }
        }

        for ( size_t fo = 0, f = 0; fo < par.firstOutSize; fo++, ifs += par.firstStride, f += par.firstStride)
        {
            size_t valueIndex          = par.secondOutSize * ( fo + ifos );
            algorithmFPType *valueBase = value + valueIndex;
            int *posBase               = selectedPos + valueIndex;

            int fUpper = f + par.firstKernelSize;
            if (fUpper > par.firstSize)
            {
                fUpper = par.firstSize;
            }
            fUpper -= f;

            for ( int fk = 0; fk < fUpper; fk++)
            {
                const algorithmFPType *dataBase = data + par.secondSize * ( fk + ifs);
                int posOffset                   = fk * (int)par.secondKernelSize;

                for (size_t ss_index = 0, so = 0; so < par.secondOutSize; so++, ss_index += par.secondStride )
                {

                    int sUpper = ss_index + par.secondKernelSize;
                    if (sUpper > par.secondSize)
                    {
                        sUpper = par.secondSize;
                        if (0 > valueBase[so])
                        {
                            valueBase[so] = 0;
                            posBase[so] = par.secondSize - ss_index;
                        }
                    }
                    sUpper -= ss_index;

                    pooling2d_opt_t<algorithmFPType,cpu> opt;
                    opt.compute_max( sUpper,
                                     posOffset,
                                     &dataBase[ss_index],
                                     &valueBase[so],
                                     &posBase[so] );

                    if (f + par.firstKernelSize > par.firstSize)
                    {
                        if (0 > valueBase[so])
                        {
                            valueBase[so] = 0;
                            posBase[so] = fUpper * (int)par.secondKernelSize;
                        }
                    }
                }
            }

        }
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesFirstZeroPaddingsCompute(
            const pooling2d::internal::Parameter &par,
            const algorithmFPType *data, algorithmFPType *value, int *selectedPos)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType minValue = -(services::internal::MaxVal<algorithmFPType>::get());

    /*
     * Loop by the first kernel dimension
     * f - index of the left upper corner of the kernel
     * fo - index of the output value
     */
    threader_for(par.firstOutSize, par.firstOutSize, [&](size_t fo)
    {
        DAAL_INT f = fo * par.firstStride;
        DAAL_INT valueIndexFirstBase = par.offsetAfter * par.secondOutSize * par.offsetBetween * fo;

        /*
         * Initialize resulting arrays
         */
        for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
        {
            DAAL_INT valueIndexSecondBase = valueIndexFirstBase + par.offsetAfter * so;
            algorithmFPType *valuePtr = value + valueIndexSecondBase;
            int *selectedPosPtr = selectedPos + valueIndexSecondBase;
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (DAAL_INT j = 0; j < par.offsetAfter; j++)
            {
                valuePtr[j] = minValue;
                selectedPosPtr[j] = -1;
            }
        }

        DAAL_INT fUpper = f + par.firstKernelSize;
        if (fUpper > par.firstSize)
        {
            fUpper = par.firstSize;
        }

        /*
         * Loops over the kernel
         */
        for (DAAL_INT fi = f; fi < fUpper; fi++)
        {
            DAAL_INT dataIndexFirstBase = par.offsetAfter * par.secondSize    * par.offsetBetween * fi;
            DAAL_INT selectedPosOffset = (fi - f) * par.secondKernelSize;
            /*
             * Loop by the second kernel dimension
             * s - index of the left upper corner of the kernel
             * so - index of the output value
             */
            for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
            {
                DAAL_INT valueIndexSecondBase = valueIndexFirstBase + par.offsetAfter * so;
                algorithmFPType *valuePtr = value + valueIndexSecondBase;
                int *selectedPosPtr = selectedPos + valueIndexSecondBase;

                DAAL_INT sUpper = s + par.secondKernelSize;
                if (sUpper > par.secondSize)
                {
                    sUpper = par.secondSize;

                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (0 > valuePtr[j])
                        {
                            valuePtr[j] = 0;
                            selectedPosPtr[j] = par.secondSize - s;
                        }
                    }
                }

                for (DAAL_INT si = s; si < sUpper; si++)
                {
                    DAAL_INT  dataIndexSecondBase =  dataIndexFirstBase + par.offsetAfter * si;
                    const algorithmFPType *dataPtr  = data  +  dataIndexSecondBase;
                    int selectedPosValue = selectedPosOffset + (si - s);

                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (dataPtr[j] > valuePtr[j])
                        {
                            valuePtr[j] = dataPtr[j];
                            selectedPosPtr[j] = selectedPosValue;
                        }
                    }
                }

                if (f + par.firstKernelSize > par.firstSize)
                {
                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (0 > valuePtr[j])
                        {
                            valuePtr[j] = 0;
                            selectedPosPtr[j] = (par.firstSize - f) * par.secondKernelSize;
                        }
                    }
                }
            }
        }
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(const pooling2d::internal::Parameter &par,
                DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                const algorithmFPType *data, algorithmFPType *valuePtr)
{
    const algorithmFPType zero = 0.0;
    algorithmFPType max = -(services::internal::MaxVal<algorithmFPType>::get());

    /*
     * Loops over the kernel
     */
    DAAL_INT fUpper = f + par.firstKernelSize;

    for (DAAL_INT fi = f; fi < fUpper; fi++)
    {
        DAAL_INT sUpper = s + par.secondKernelSize;

        PRAGMA_NOVECTOR
        for (DAAL_INT si = s; si < sUpper; si++)
        {
            DAAL_INT dataIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));

            bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));
            algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);
            if (dataValue > max)
            {
                max = dataValue;
            }
        }
    }
    valuePtr[j] = max;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesLastZeroPaddingsCompute(
                const pooling2d::internal::Parameter &par,
                const algorithmFPType *data, algorithmFPType *value)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType minValue = -(services::internal::MaxVal<algorithmFPType>::get());
    int iFirstKernelSize = (int)par.firstKernelSize;
    int iSecondKernelSize = (int)par.secondKernelSize;
    threader_for(par.offsetBefore, par.offsetBefore, [&](size_t i)
    {
        /*
         * Initialize resulting arrays
         */
        for (DAAL_INT fo = 0; fo < par.firstOutSize; fo++)
        {
            DAAL_INT valueIndex = par.secondOutSize * (fo + par.firstOutSize * i);
            for (DAAL_INT so = 0; so < par.secondOutSize; so++, valueIndex++)
            {
                value[valueIndex] = minValue;
            }
        }

        for (DAAL_INT f = 0, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            /*
             * Loop over the first kernel dimension
             */
            DAAL_INT fUpper = f + par.firstKernelSize;
            if (fUpper > par.firstSize)
            {
                fUpper = par.firstSize;
            }
            int fi = 0;
            for (DAAL_INT ffi = f; ffi < fUpper; fi++, ffi++)
            {
                /*
                 * Resulting value index
                 */
                DAAL_INT valueIndex = par.secondOutSize * (fo + par.firstOutSize * i);
                DAAL_INT so = 0, s = 0;
                const algorithmFPType *dataPtrBase = data + par.secondSize * (ffi + par.firstSize * i);

                for (; so < par.secondOutSize; so++, valueIndex++, s += par.secondStride)
                {
                    /*
                     * Input data index
                     */
                    const algorithmFPType * const dataPtr = dataPtrBase + s;

                    DAAL_INT sUpper = s + par.secondKernelSize;
                    if (sUpper > par.secondSize)
                    {
                        sUpper = par.secondSize;
                        if (0 > value[valueIndex])
                        {
                            value[valueIndex] = 0;
                        }
                    }
                    sUpper -= s;
                    algorithmFPType max = value[valueIndex];

                    /*
                     * Loop over the second kernel dimension
                    */
                    PRAGMA_NOVECTOR
                    for (int si = 0; si < sUpper; si++)
                    {
                        if (dataPtr[si] > max)
                        {
                            max = dataPtr[si];
                        }
                    }

                    value[valueIndex] = max;

                    if (f + par.firstKernelSize > par.firstSize)
                    {
                        if (0 > value[valueIndex])
                        {
                            value[valueIndex] = 0;
                        }
                    }
                }
            }
        }
    } );
}
template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::indicesFirstZeroPaddingsCompute(
            const pooling2d::internal::Parameter &par,
            const algorithmFPType *data, algorithmFPType *value)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType minValue = -(services::internal::MaxVal<algorithmFPType>::get());

    /*
     * Loop by the first kernel dimension
     * f - index of the left upper corner of the kernel
     * fo - index of the output value
     */
    threader_for(par.firstOutSize, par.firstOutSize, [&](size_t fo)
    {
        DAAL_INT f = fo * par.firstStride;
        DAAL_INT valueIndexFirstBase = par.offsetAfter * par.secondOutSize * par.offsetBetween * fo;

        /*
         * Initialize resulting arrays
         */
        for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
        {
            DAAL_INT valueIndexSecondBase = valueIndexFirstBase + par.offsetAfter * so;
            algorithmFPType *valuePtr = value + valueIndexSecondBase;
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (DAAL_INT j = 0; j < par.offsetAfter; j++)
            {
                valuePtr[j] = minValue;
            }
        }

        DAAL_INT fUpper = f + par.firstKernelSize;
        if (fUpper > par.firstSize)
        {
            fUpper = par.firstSize;
        }

        /*
         * Loops over the kernel
         */
        for (DAAL_INT fi = f; fi < fUpper; fi++)
        {
            DAAL_INT dataIndexFirstBase = par.offsetAfter * par.secondSize    * par.offsetBetween * fi;
            /*
             * Loop by the second kernel dimension
             * s - index of the left upper corner of the kernel
             * so - index of the output value
             */
            for (DAAL_INT s = 0, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
            {
                DAAL_INT valueIndexSecondBase = valueIndexFirstBase + par.offsetAfter * so;
                algorithmFPType *valuePtr = value + valueIndexSecondBase;
                DAAL_INT sUpper = s + par.secondKernelSize;
                if (sUpper > par.secondSize)
                {
                    sUpper = par.secondSize;

                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (0 > valuePtr[j])
                        {
                            valuePtr[j] = 0;
                        }
                    }
                }

                for (DAAL_INT si = s; si < sUpper; si++)
                {
                    DAAL_INT  dataIndexSecondBase =  dataIndexFirstBase + par.offsetAfter * si;
                    const algorithmFPType *dataPtr  = data  +  dataIndexSecondBase;

                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (dataPtr[j] > valuePtr[j])
                        {
                            valuePtr[j] = dataPtr[j];
                        }
                    }
                }

                if (f + par.firstKernelSize > par.firstSize)
                {
                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
                    {
                        if (0 > valuePtr[j])
                        {
                            valuePtr[j] = 0;
                        }
                    }
                }
            }
        }
    } );
}

} // namespace internal
} // namespace forward
} // namespace maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
