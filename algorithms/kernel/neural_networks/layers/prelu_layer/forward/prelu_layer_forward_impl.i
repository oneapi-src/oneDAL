/* file: prelu_layer_forward_impl.i */
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

#ifndef __PRELU_LAYER_FORWARD_IMPL_I__
#define __PRELU_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PReLUKernel<algorithmFPType, method, cpu>::compute(const prelu::forward::Input *input, const prelu::Parameter *parameter,
                                                        prelu::forward::Result *result)
{
    size_t wStart = parameter->dataDimension;
    size_t wLen   = parameter->weightsDimension;

    SharedPtr<Tensor> inputTensor   = input->get(layers::forward::data);
    SharedPtr<Tensor> weightsTensor = input->get(layers::forward::weights);
    SharedPtr<Tensor> resultTensor  = result->get(layers::forward::value);

    const services::Collection<size_t> &inDims = inputTensor->getDimensions();
    const services::Collection<size_t> &wDims  = weightsTensor->getDimensions();

    size_t nDims = inDims.size();

    size_t *dimsCounter = (size_t *)services::daal_malloc(sizeof(size_t) * nDims);
    if(!dimsCounter) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, inDims[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    weightsTensor->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, inDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t size = 1;
    for(size_t i = 0; i < nDims; i++)
    {
        dimsCounter[i] = 0;
        size *= inDims[i];
    }

    for(size_t i = 0; i < size; i++)
    {
        size_t wJ = 0;
        size_t mul = 1;

        for(size_t d = wStart + wLen; d >= wStart + 1; d--)
        {
            wJ += dimsCounter[d - 1] * mul;
            mul *= inDims[d - 1];
        }

        if (inputArray[i] >= (algorithmFPType)0)
        {
            resultArray[i] = inputArray[i];
        }
        else
        {
            resultArray[i] = inputArray[i] * wArray[wJ];
        }

        for(size_t d = 1; d < nDims + 1; d++)
        {
            dimsCounter[nDims - d]++;
            if(dimsCounter[nDims - d] < inDims[nDims - d]) { break; }
            dimsCounter[nDims - d] = 0;
        }
    }

    inputTensor->releaseSubtensor(inputBlock);
    weightsTensor->releaseSubtensor(wBlock);
    resultTensor->releaseSubtensor(resultBlock);

    services::daal_free(dimsCounter);
}

} // namespace internal
} // namespace forward
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
