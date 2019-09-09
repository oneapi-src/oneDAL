/* file: eltwise_sum_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of element-wise sum algorithm
//--
*/

#ifndef __ELTWISE_SUM_LAYER_BACKWARD_IMPL_I__
#define __ELTWISE_SUM_LAYER_BACKWARD_IMPL_I__

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
namespace eltwise_sum
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status EltwiseSumKernel<algorithmFPType, method, cpu>::compute(
    Tensor *inputGradient, Tensor *coefficients, Tensor **outputs, size_t nOutputs)
{
    if (checkForInPlace(inputGradient, coefficients, outputs, nOutputs))
    {
        return Status();
    }

    __DAAL_MAKE_TENSOR_THREADSAFE(inputGradient);
    for (size_t i = 0; i < nOutputs; i++)
    {
        __DAAL_MAKE_TENSOR_THREADSAFE(outputs[i]);
    }

    algorithmFPType *coefficientsArray = nullptr;
    ReadSubtensor<algorithmFPType, cpu, Tensor> coefficientsBlock;
    if (coefficients)
    {
        DAAL_ASSERT(nOutputs == coefficients->getDimensionSize(0));

        coefficientsBlock.set(*coefficients, 0, 0, 0, nOutputs);
        DAAL_CHECK_BLOCK_STATUS(coefficientsBlock);

        coefficientsArray = const_cast<algorithmFPType *>(coefficientsBlock.get());
    }

    SafeStatus safeStat;
    daal::threader_for(nOutputs, nOutputs, [ =, &safeStat ](size_t i)
    {
        safeStat |= processOutputTensor(inputGradient, coefficientsArray, outputs[i], i);
    });
    if (!safeStat) { return safeStat.detach(); }

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EltwiseSumKernel<algorithmFPType, method, cpu>::processOutputTensor(
    Tensor *inputGradient, const algorithmFPType *coefficientsArray, Tensor *output, size_t outputIndex)
{
    return computeImpl<cpu>(*output, [ = ](size_t fDimN, size_t *fDims, size_t dimensionSize, const TensorOffsetLayout &layout) -> Status
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(*inputGradient, fDimN, fDims, 0, dimensionSize, layout);
        WriteSubtensor<algorithmFPType, cpu, Tensor> outputBlock(*output, fDimN, fDims, 0, dimensionSize, layout);

        DAAL_CHECK_BLOCK_STATUS(inputGradientBlock);
        DAAL_CHECK_BLOCK_STATUS(outputBlock);

        const algorithmFPType *inputGradientArray = inputGradientBlock.get();
        algorithmFPType *outputArray              = outputBlock.get();
        const size_t inputGradientBlockSize       = inputGradientBlock.getSize();
        const size_t outputBlockSize              = outputBlock.getSize();

        DAAL_ASSERT(inputGradientBlockSize == outputBlockSize);

        if (coefficientsArray)
        {
            const algorithmFPType coefficient = coefficientsArray[outputIndex];

          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < outputBlockSize; i++)
            {
                outputArray[i] = coefficient * inputGradientArray[i];
            }
        }
        else
        {
            for (size_t i = 0; i < outputBlockSize; i++)
            {
                outputArray[i] = inputGradientArray[i];
            }
        }
        return Status();
    });
}

template<typename algorithmFPType, Method method, CpuType cpu>
bool EltwiseSumKernel<algorithmFPType, method, cpu>::checkForInPlace(
    const Tensor *inputGradient, const Tensor *coefficients, Tensor **outputs, size_t nOutputs)
{
    if (coefficients) { return false; }
    for (size_t i = 0; i < nOutputs; i++)
    {
        if (outputs[i] != inputGradient) { return false; }
    }
    return true;
}

} // internal
} // backward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
