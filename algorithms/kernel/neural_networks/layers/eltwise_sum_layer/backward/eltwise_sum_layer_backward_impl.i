/* file: eltwise_sum_layer_backward_impl.i */
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
