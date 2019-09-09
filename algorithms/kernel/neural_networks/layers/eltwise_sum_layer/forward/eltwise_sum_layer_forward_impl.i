/* file: eltwise_sum_layer_forward_impl.i */
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

#ifndef __ELTWISE_SUM_LAYER_FORWARD_IMPL_I__
#define __ELTWISE_SUM_LAYER_FORWARD_IMPL_I__

using namespace daal::internal;
using namespace daal::services;

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
inline void computeInternalSum(const algorithmFPType *inputArray, algorithmFPType *valueArray, size_t blockSize,
                               const algorithmFPType *coefficientsArray, size_t inputIndex)
{
    const algorithmFPType alpha = coefficientsArray[inputIndex];

    if (inputIndex > 0)
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < blockSize; i++)
        {
            valueArray[i] += alpha * inputArray[i];
        }
    }
    else
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < blockSize; i++)
        {
            valueArray[i] = alpha * inputArray[i];
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
inline void computeInternalSum(const algorithmFPType *inputArray, algorithmFPType *valueArray,
                               size_t blockSize, size_t inputIndex)
{
    if (inputIndex > 0)
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < blockSize; i++)
        {
            valueArray[i] += inputArray[i];
        }
    }
    else
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            valueArray[i] = inputArray[i];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EltwiseSumKernel<algorithmFPType, method, cpu>::compute(Tensor **inputs, Tensor *value,
    Tensor *coefficients, Tensor *auxCoefficients, NumericTable *numberOfCoefficients, size_t nInputs)
{
    Status s;
    DAAL_CHECK_STATUS(s, makeResultForBackward(coefficients, auxCoefficients, numberOfCoefficients, nInputs));

    if (coefficients)
    {
        DAAL_ASSERT(nInputs == coefficients->getDimensionSize(0));

        ReadSubtensor<algorithmFPType, cpu, Tensor> coefficientsBlock(*coefficients, 0, 0, 0, nInputs);
        DAAL_CHECK_BLOCK_STATUS(coefficientsBlock);

        const algorithmFPType *coefficientsArray = coefficientsBlock.get();

        DAAL_CHECK_STATUS(s, computeGeneric(inputs, value, coefficientsArray, nInputs));
    }
    else
    {
        DAAL_CHECK_STATUS(s, computeGeneric(inputs, value, nullptr, nInputs));
    }

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EltwiseSumKernel<algorithmFPType, method, cpu>::computeGeneric(
    Tensor **inputs, Tensor *value, const algorithmFPType *coefficients, size_t nInputs)
{
    for (size_t i = 0; i < nInputs; i++)
    {
        __DAAL_MAKE_TENSOR_THREADSAFE(inputs[i]);
    }

    return computeImpl<cpu>(*value, [ = ](size_t fDimN, size_t *fDims, size_t dimensionSize, const TensorOffsetLayout &layout) -> Status
    {
        WriteSubtensor<algorithmFPType, cpu, Tensor> valueBlock(*value, fDimN, fDims, 0, dimensionSize, layout);
        DAAL_CHECK_BLOCK_STATUS(valueBlock);

        algorithmFPType *valueArray = valueBlock.get();
        const size_t valueBlockSize = valueBlock.getSize();

        for (size_t inputIndex = 0; inputIndex < nInputs; inputIndex++)
        {
            Tensor *inputTensor = inputs[inputIndex];
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, fDimN, fDims, 0, dimensionSize, layout);
            DAAL_CHECK_BLOCK_STATUS(inputBlock);

            const algorithmFPType *inputArray = inputBlock.get();
            const size_t inputBlockSize = valueBlock.getSize();

            DAAL_ASSERT(inputBlockSize == valueBlockSize);

            if (coefficients)
            {
                computeInternalSum<algorithmFPType, cpu>(
                    inputArray, valueArray, inputBlockSize, coefficients, inputIndex);
            }
            else
            {
                computeInternalSum<algorithmFPType, cpu>(
                    inputArray, valueArray, inputBlockSize, inputIndex);
            }
        }
        return Status();
    });
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EltwiseSumKernel<algorithmFPType, method, cpu>::makeResultForBackward(
    Tensor *coefficients, Tensor *auxCoefficients, NumericTable *numberOfCoefficients, size_t nInputs)
{
    if (coefficients)
    {
        DAAL_ASSERT(auxCoefficients);

        if (coefficients != auxCoefficients)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> coefficientsBlock(*coefficients, 0, 0, 0, nInputs);
            WriteSubtensor<algorithmFPType, cpu, Tensor> auxCoefficientsBlock(*auxCoefficients, 0, 0, 0, nInputs);

            DAAL_CHECK_BLOCK_STATUS(coefficientsBlock);
            DAAL_CHECK_BLOCK_STATUS(auxCoefficientsBlock);

            const algorithmFPType *coefficientsArray = coefficientsBlock.get();
            algorithmFPType *auxCoefficientsArray = auxCoefficientsBlock.get();

            for (size_t i = 0; i < nInputs; i++)
            {
                auxCoefficientsArray[i] = coefficientsArray[i];
            }
        }
    }
    else
    {
        DAAL_ASSERT(numberOfCoefficients);
        DAAL_ASSERT(numberOfCoefficients->getNumberOfRows() == 1);

        WriteRows<int, cpu, NumericTable> numberOfCoefficientsBlock(numberOfCoefficients, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(numberOfCoefficientsBlock);

        int *numberOfCoefficientsPtr = numberOfCoefficientsBlock.get();
        *numberOfCoefficientsPtr = (int)nInputs;
    }

    return Status();
}

} // internal
} // forward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
