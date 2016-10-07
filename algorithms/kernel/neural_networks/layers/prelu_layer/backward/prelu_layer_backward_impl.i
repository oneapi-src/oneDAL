/* file: prelu_layer_backward_impl.i */
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
//  Implementation of prelu calculation functions.
//--
*/

#ifndef __PRELU_LAYER_BACKWARD_IMPL_I__
#define __PRELU_LAYER_BACKWARD_IMPL_I__

#include "service_tensor.h"
#include "service_numeric_table.h"

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
namespace prelu
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PReLUKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor,
                                                        Tensor *wDerTensor, Tensor *resultTensor, const prelu::Parameter *parameter)
{
    size_t wStart = parameter->dataDimension;
    size_t wLen   = parameter->weightsDimension;

    const services::Collection<size_t> &xDims = xTensor->getDimensions();
    const services::Collection<size_t> &wDims = wDerTensor->getDimensions();

    size_t nDims = xDims.size();

    TSmartPtr<size_t, cpu> dimsCounterPtr(nDims);
    size_t *dimsCounter = dimsCounterPtr.get();
    if(!dimsCounter) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    ReadSubtensor<algorithmFPType, cpu> inGradBlock(inGradTensor, 0, 0, 0, xDims[0]);
    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock(xTensor, 0, 0, 0, xDims[0]);
    const algorithmFPType *xArray = xBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> wDerBlock(wDerTensor, 0, 0, 0, wDims[0]);
    algorithmFPType *wDerArray = wDerBlock.get();

    size_t size = xTensor->getSize();

    for(size_t i = 0; i < nDims; i++)
    {
        dimsCounter[i] = (size_t)0;
    }

    size_t wSize = wDerTensor->getSize();
    for(size_t i = 0; i < wSize; i++)
    {
        wDerArray[i] = (algorithmFPType)0;
    }

    for(size_t i = 0; i < size; i++)
    {
        size_t wJ = 0;
        size_t mul = 1;

        for(size_t d = wStart + wLen; d >= wStart + 1; d--)
        {
            wJ += dimsCounter[d - 1] * mul;
            mul *= xDims[d - 1];
        }

        if (xArray[i] < (algorithmFPType)0)
        {
            wDerArray[wJ] += inGradArray[i] * xArray[i];
        }

        for(size_t d = 1; d < nDims + 1; d++)
        {
            dimsCounter[nDims - d]++;
            if(dimsCounter[nDims - d] < xDims[nDims - d]) { break; }
            dimsCounter[nDims - d] = 0;
        }
    }

    if (parameter->propagateGradient)
    {
        ReadSubtensor<algorithmFPType, cpu> wBlock(wTensor, 0, 0, 0, wDims[0]);
        const algorithmFPType *wArray = wBlock.get();

        WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, xDims[0]);
        algorithmFPType *resultArray = resultBlock.get();

        for(size_t i = 0; i < nDims; i++)
        {
            dimsCounter[i] = (size_t)0;
        }
        for(size_t i = 0; i < size; i++)
        {
            size_t wJ = 0;
            size_t mul = 1;

            for(size_t d = wStart + wLen; d >= wStart + 1; d--)
            {
                wJ += dimsCounter[d - 1] * mul;
                mul *= xDims[d - 1];
            }

            if (xArray[i] == (algorithmFPType)0)
            {
                resultArray[i] = (algorithmFPType)0;
            }
            else if (xArray[i] > (algorithmFPType)0)
            {
                resultArray[i] = inGradArray[i];
            }
            else
            {
                resultArray[i] = inGradArray[i] * wArray[wJ];
            }

            for(size_t d = 1; d < nDims + 1; d++)
            {
                dimsCounter[nDims - d]++;
                if(dimsCounter[nDims - d] < xDims[nDims - d]) { break; }
                dimsCounter[nDims - d] = 0;
            }
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
