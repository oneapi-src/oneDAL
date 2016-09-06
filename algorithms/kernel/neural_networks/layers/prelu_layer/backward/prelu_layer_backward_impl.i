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

using namespace daal::services;

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
void PReLUKernel<algorithmFPType, method, cpu>::compute(const prelu::backward::Input *input, const prelu::Parameter *parameter,
                                                        prelu::backward::Result *result)
{
    size_t wStart = parameter->dataDimension;
    size_t wLen   = parameter->weightsDimension;

    SharedPtr<Tensor> inGradTensor  = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> xTensor       = input->get(prelu::auxData);
    SharedPtr<Tensor> wTensor       = input->get(prelu::auxWeights);
    SharedPtr<Tensor> wDerTensor    = result->get(layers::backward::weightDerivatives);
    SharedPtr<Tensor> resultTensor  = result->get(layers::backward::gradient);

    const services::Collection<size_t> &xDims = xTensor->getDimensions();
    const services::Collection<size_t> &wDims = wDerTensor->getDimensions();

    size_t nDims = xDims.size();

    size_t *dimsCounter = (size_t *)services::daal_malloc(sizeof(size_t) * nDims);
    if(!dimsCounter) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inGradBlock;
    inGradTensor->getSubtensor(0, 0, 0, xDims[0], readOnly, inGradBlock);
    algorithmFPType *inGradArray = inGradBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> xBlock;
    xTensor->getSubtensor(0, 0, 0, xDims[0], readOnly, xBlock);
    algorithmFPType *xArray = xBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTensor->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wDerBlock;
    wDerTensor->getSubtensor(0, 0, 0, wDims[0], writeOnly, wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, xDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t wSize = wBlock.getSize();

    size_t size = 1;
    for(size_t i = 0; i < nDims; i++)
    {
        dimsCounter[i] = (size_t)0;
        size *= xDims[i];
    }

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
            wDerArray[wJ] += inGradArray[i] * xArray[i];
        }

        for(size_t d = 1; d < nDims + 1; d++)
        {
            dimsCounter[nDims - d]++;
            if(dimsCounter[nDims - d] < xDims[nDims - d]) { break; }
            dimsCounter[nDims - d] = 0;
        }
    }

    inGradTensor->releaseSubtensor(inGradBlock);
    xTensor->releaseSubtensor(xBlock);
    wTensor->releaseSubtensor(wBlock);
    wDerTensor->releaseSubtensor(wDerBlock);
    resultTensor->releaseSubtensor(resultBlock);

    services::daal_free(dimsCounter);
}

} // namespace internal
} // namespace backward
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
