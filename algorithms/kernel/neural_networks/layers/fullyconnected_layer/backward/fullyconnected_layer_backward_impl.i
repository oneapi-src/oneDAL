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
void FullyconnectedKernel<algorithmFPType, method, cpu>::compute(const fullyconnected::backward::Input *input,
    const fullyconnected::Parameter *parameter, fullyconnected::backward::Result *result)
{
    SharedPtr<Tensor> inGradTable  = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> xTable       = input->get(fullyconnected::auxData);
    SharedPtr<Tensor> wTable       = input->get(fullyconnected::auxWeights);
    SharedPtr<Tensor> wDerTable    = result->get(layers::backward::weightDerivatives);
    SharedPtr<Tensor> bDerTable    = result->get(layers::backward::biasDerivatives);
    SharedPtr<Tensor> resultTable  = result->get(layers::backward::gradient);

    size_t m = parameter->nOutputs;

    const services::Collection<size_t>& xDims = xTable->getDimensions();
    const services::Collection<size_t>& wDims = wDerTable->getDimensions();

    size_t nDims = xDims.size();

    size_t* dimsCounter = (size_t*)services::daal_malloc(sizeof(size_t) * nDims);
    if(!dimsCounter) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inGradBlock;
    inGradTable->getSubtensor(0, 0, 0, xDims[0], readOnly, inGradBlock);
    algorithmFPType *inGradArray = inGradBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> xBlock;
    xTable->getSubtensor(0, 0, 0, xDims[0], readOnly, xBlock);
    algorithmFPType *xArray = xBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTable->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wDerBlock;
    wDerTable->getSubtensor(0, 0, 0, wDims[0], writeOnly, wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bDerBlock;
    bDerTable->getSubtensor(0, 0, 0, m, writeOnly, bDerBlock);
    algorithmFPType *bDerArray = bDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, xDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t size = 1;
    for(size_t i=0; i<nDims; i++)
    {
        dimsCounter[i] = 0;
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

    {
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
            &lda, inGradArray, &ldb, &beta, resultArray, &ldc);
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

        Blas<algorithmFPType, cpu>::xgemm(&transa, &transb, &_m, &_n, &_k, &alpha, xArray,
            &lda, inGradArray, &ldb, &beta, wDerArray, &ldc);
    }

    inGradTable->releaseSubtensor(inGradBlock);
    xTable->releaseSubtensor(xBlock);
    wTable->releaseSubtensor(wBlock);
    wDerTable->releaseSubtensor(wDerBlock);
    bDerTable->releaseSubtensor(bDerBlock);
    resultTable->releaseSubtensor(resultBlock);

    services::daal_free( dimsCounter );
}

} // internal
} // backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
