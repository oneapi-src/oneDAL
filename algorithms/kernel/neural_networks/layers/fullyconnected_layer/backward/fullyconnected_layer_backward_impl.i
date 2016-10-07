/* file: fullyconnected_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of fullyconnected algorithm
//--
*/

#ifndef __FULLYCONNECTED_LAYER_BACKWARD_IMPL_I__
#define __FULLYCONNECTED_LAYER_BACKWARD_IMPL_I__

#include "threading.h"
#include "service_blas.h"
#include "service_tensor.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void FullyconnectedKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor, Tensor *wDerTensor,
                                                                 Tensor *bDerTensor, Tensor *resultTensor, const fullyconnected::Parameter *parameter)
{
    size_t m = parameter->nOutputs;

    const services::Collection<size_t>& xDims = xTensor->getDimensions();
    const services::Collection<size_t>& wDims = wDerTensor->getDimensions();

    size_t nDims = xDims.size();

    ReadSubtensor<algorithmFPType, cpu> inGradBlock(inGradTensor, 0, 0, 0, xDims[0]);
    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock(xTensor, 0, 0, 0, xDims[0]);
    const algorithmFPType *xArray = xBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> wDerBlock(wDerTensor, 0, 0, 0, wDims[0]);
    algorithmFPType *wDerArray = wDerBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> bDerBlock(bDerTensor, 0, 0, 0, m);
    algorithmFPType *bDerArray = bDerBlock.get();

    size_t size = 1;
    for(size_t i=0; i<nDims; i++)
    {
        size *= xDims[i];
    }
    size_t wSize = size / xDims[0] * m;

    for(size_t i=0; i<m; i++)
    {
        bDerArray[i] = (algorithmFPType)0.0;
    }

    algorithmFPType invBatchSize = 1.0 / xDims[0];

    if( xDims[0]>10 && m>100 )
    {
        daal::threader_for( xDims[0], xDims[0], [=](int j)
        {
            for(size_t i=0; i<m; i++)
            {
                bDerArray[i] += inGradArray[j*m + i] * invBatchSize;
            }
        } );
    }
    else
    {
        for(size_t j=0; j<xDims[0]; j++)
        {
            for(size_t i=0; i<m; i++)
            {
                bDerArray[i] += inGradArray[j*m + i] * invBatchSize;
            }
        }
    }

    typedef typename Blas<algorithmFPType, cpu>::SizeType BlasSize;

    if (parameter->propagateGradient)
    {
        ReadSubtensor<algorithmFPType, cpu> wBlock(wTensor, 0, 0, 0, wDims[0]);
        algorithmFPType *wArray = const_cast<algorithmFPType*>(wBlock.get());

        WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, xDims[0]);
        algorithmFPType *resultArray = resultBlock.get();

        char transa = 'n';
        char transb = 'n';
        BlasSize _m = size/xDims[0];
        BlasSize _n = xDims[0];
        BlasSize _k = m;
        algorithmFPType alpha = 1.0;
        BlasSize lda = size/xDims[0];
        BlasSize ldb = m;
        algorithmFPType beta = 0.0;
        BlasSize ldc = size/xDims[0];

        Blas<algorithmFPType, cpu>::xgemm(&transa, &transb, &_m, &_n, &_k, &alpha, wArray,
            &lda, const_cast<algorithmFPType*>(inGradArray), &ldb, &beta, resultArray, &ldc);
    }

    {
        char transa = 'n';
        char transb = 't';
        BlasSize _m = size/xDims[0];
        BlasSize _n = m;
        BlasSize _k = xDims[0];
        algorithmFPType alpha = invBatchSize;
        BlasSize lda = size/xDims[0];
        BlasSize ldb = m;
        algorithmFPType beta = 0.0;
        BlasSize ldc = size/xDims[0];

        Blas<algorithmFPType, cpu>::xgemm(&transa, &transb, &_m, &_n, &_k, &alpha, const_cast<algorithmFPType*>(xArray),
            &lda, const_cast<algorithmFPType*>(inGradArray), &ldb, &beta, wDerArray, &ldc);
    }
}

} // internal
} // backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
