/* file: truncated_gaussian_impl.i */
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
//  Implementation of truncated gaussian initializer
//--
*/
#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_IMPL_I__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_IMPL_I__

#include "initializers_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace internal
{
/* sqrt(2) */
template<typename algorithmFPType> inline algorithmFPType sqrt_2        (void){ return algorithmFPType(0.0); }
template<>                         inline float           sqrt_2<float> (void){ uint32_t iv = 0x3FB504F3; float v = *(float*)&iv; return v; }
template<>                         inline double          sqrt_2<double>(void){ uint64_t iv = 0x3FF6A09E667F3BCD; double v = *(double*)&iv; return v; }

using namespace daal::algorithms::distributions::uniform::internal;

/* result = sigma * [ ICDF( CDF(a) + uniform(0,1) * ( CDF(b) - CDF(a)) ) ] + mean;
    CDF(a) = 0, if a = -inf;
    CDF(b) = 1, if b = +inf;
*/
template<typename algorithmFPType, Method method, CpuType cpu>
Status TruncatedGaussianKernel<algorithmFPType, method, cpu>::compute(
    const TruncatedGaussianInitializerTaskDescriptor<algorithmFPType> &desc)
{
    initializers::internal::EngineImpl<cpu> engine(desc.engine);
    DAAL_CHECK_MALLOC(engine.get());

    Tensor *resultTensor = desc.result;
    size_t size = resultTensor->getSize();

    algorithmFPType mean     = (algorithmFPType)desc.mean;
    algorithmFPType sigma    = (algorithmFPType)desc.sigma;
    algorithmFPType cdf_b    = getCDFNormal(desc.b, mean, sigma);
    algorithmFPType cdf_a    = getCDFNormal(desc.a, mean, sigma);
    algorithmFPType cdf_diff = cdf_b - cdf_a;

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, 0, resultTensor->getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    Status s;
    DAAL_CHECK_STATUS(s, (UniformKernelDefault<algorithmFPType, cpu>::compute((algorithmFPType)0.0, (algorithmFPType)1.0,
                                                                              *engine, size, resultArray)));

    size_t nBlocks = size / _nElemsInBlock;
    nBlocks += (nBlocks * _nElemsInBlock != size);

    daal::threader_for(nBlocks, nBlocks, [ & ](int block)
    {
        size_t nElemsToProcess = _nElemsInBlock;
        size_t shift = block * _nElemsInBlock;

        algorithmFPType *resultLocal = &resultArray[shift];

        if( block == nBlocks - 1 )
        {
            nElemsToProcess = size - shift;
        }

        for(size_t i = 0; i < nElemsToProcess; i++)
        {
            resultLocal[i] = cdf_a + resultLocal[i] * cdf_diff;
        }

        Math<algorithmFPType,cpu>::vCdfNormInv(nElemsToProcess, resultLocal, resultLocal);

        for(size_t i = 0; i < nElemsToProcess; i++)
        {
            resultLocal[i] = sigma * resultLocal[i] + mean;
        }
    } );

    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
algorithmFPType TruncatedGaussianKernel<algorithmFPType, method, cpu>::getCDFNormal(
    algorithmFPType p, algorithmFPType mean, algorithmFPType sigma)
{
    return (algorithmFPType)0.5 * ( (algorithmFPType)1.0 + Math<algorithmFPType, cpu>::sErf(
        (p - mean) / (sigma * sqrt_2<algorithmFPType>() )) );
}

} // internal
} // namespace truncated_gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
