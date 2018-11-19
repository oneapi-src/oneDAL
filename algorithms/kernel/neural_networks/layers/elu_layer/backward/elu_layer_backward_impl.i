/* file: elu_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of ELU algorithm
//--
*/

#ifndef __ELU_LAYER_BACKWARD_IMPL_I__
#define __ELU_LAYER_BACKWARD_IMPL_I__

#include "elu_common.h"

#include "service_math.h"
#include "service_tensor.h"
#include "service_mkl_tensor.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace backward
{
namespace internal
{

using namespace daal::internal;

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::compute(const Parameter &parameter,
                                                        const Tensor &inputGradientTensor,
                                                        const Tensor &auxDataTensor,
                                                        const Tensor *auxValueTensor,
                                                              Tensor &gradientTensor)
{
    if (!auxValueTensor)
    {
        const algorithmFPType alpha = parameter.alpha;
        return computeWithoutAuxValues(inputGradientTensor, auxDataTensor, gradientTensor, alpha);
    }
    else
    {
        if (elu::internal::canComputeInMklLayout<algorithmFPType, cpu>(auxDataTensor,
                                                                       inputGradientTensor,
                                                                       gradientTensor))
        {
            return computeInMklLayout(inputGradientTensor, auxDataTensor, *auxValueTensor, gradientTensor);
        }
        else
        {
            return computeLayoutAgnostic(inputGradientTensor, auxDataTensor, *auxValueTensor, gradientTensor);
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::computeLayoutAgnostic(const Tensor &inputGradientTensor,
                                                                      const Tensor &auxDataTensor,
                                                                      const Tensor &auxValueTensor,
                                                                            Tensor &gradientTensor)
{
    ReadSubtensor<algorithmFPType, cpu> inputGradientBlock(const_cast<Tensor &>(inputGradientTensor));
    DAAL_CHECK_BLOCK_STATUS(inputGradientBlock);

    ReadSubtensor<algorithmFPType, cpu> auxDataBlock(const_cast<Tensor &>(auxDataTensor));
    DAAL_CHECK_BLOCK_STATUS(auxDataBlock);

    ReadSubtensor<algorithmFPType, cpu> auxValueBlock(const_cast<Tensor &>(auxValueTensor));
    DAAL_CHECK_BLOCK_STATUS(auxValueBlock);

    WriteSubtensor<algorithmFPType, cpu> gradientBlock(gradientTensor);
    DAAL_CHECK_BLOCK_STATUS(gradientBlock);

    computeInRawLayout(inputGradientBlock.get(),
                       auxDataBlock.get(),
                       auxValueBlock.get(),
                       gradientBlock.get(),
                       inputGradientBlock.getSize());

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::computeInMklLayout(const Tensor &inputGradientTensor,
                                                                   const Tensor &auxDataTensor,
                                                                   const Tensor &auxValueTensor,
                                                                         Tensor &gradientTensor)
{
    using MklTensorType = MklTensor<algorithmFPType>;

    /* We assume tensros can be casted and check was performed by the caller of this function */
    auto &inputGradientMklTensor = const_cast<MklTensorType &>(static_cast<const MklTensorType &>(inputGradientTensor));
    auto &auxDataMklTensor       = const_cast<MklTensorType &>(static_cast<const MklTensorType &>(auxDataTensor));
    auto &gradientMklTensor      = static_cast<MklTensorType &>(gradientTensor);
    gradientMklTensor.setDnnLayout(inputGradientMklTensor.getSharedDnnLayout());

    ReadSubtensor<algorithmFPType, cpu> auxValueBlock(const_cast<Tensor &>(auxValueTensor));
    DAAL_CHECK_BLOCK_STATUS(auxValueBlock);

    computeInRawLayout(inputGradientMklTensor.getDnnArray(),
                       auxDataMklTensor.getDnnArray(),
                       auxValueBlock.get(),
                       gradientMklTensor.getDnnArray(),
                       inputGradientMklTensor.getSize());

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeInRawLayout(const algorithmFPType *inputGradient,
                                                                 const algorithmFPType *auxData,
                                                                 const algorithmFPType *auxValue,
                                                                       algorithmFPType *gradient,
                                                                       size_t dataSize)
{
    elu::internal::computeThreaded<algorithmFPType, cpu>(dataSize,
    [ & ](size_t offset, size_t blockSize)
    {
        computeBlock(inputGradient + offset, auxData + offset,
                     auxValue + offset, gradient + offset, blockSize);
    });
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeBlock(const algorithmFPType *inputGradient,
                                                           const algorithmFPType *auxData,
                                                           const algorithmFPType *auxValue,
                                                                 algorithmFPType *gradient,
                                                                 size_t blockSize)
{
    BlockSizeType auxValueCounter = 0;
    for (BlockSizeType i = 0; i < blockSize; ++i)
    {
        if (auxData[i] > (algorithmFPType)0.0)
        {
            gradient[i] = inputGradient[i];
        }
        else
        {
            gradient[i] = inputGradient[i] * auxValue[auxValueCounter++];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::computeWithoutAuxValues(const Tensor &inputGradientTensor,
                                                                        const Tensor &auxDataTensor,
                                                                              Tensor &gradientTensor,
                                                                              algorithmFPType alpha)
{
    ReadSubtensor<algorithmFPType, cpu> inputGradientBlock(const_cast<Tensor &>(inputGradientTensor));
    DAAL_CHECK_BLOCK_STATUS(inputGradientBlock);

    ReadSubtensor<algorithmFPType, cpu> auxDataBlock(const_cast<Tensor &>(auxDataTensor));
    DAAL_CHECK_BLOCK_STATUS(auxDataBlock);

    WriteSubtensor<algorithmFPType, cpu> gradientBlock(gradientTensor);
    DAAL_CHECK_BLOCK_STATUS(gradientBlock);

    const algorithmFPType *inputGradient = inputGradientBlock.get();
    const algorithmFPType *auxData       = auxDataBlock.get();
          algorithmFPType *gradient      = gradientBlock.get();

    elu::internal::computeThreaded<algorithmFPType, cpu>(inputGradientTensor.getSize(),
    [ & ](size_t offset, size_t blockSize)
    {
        computeBlockWithoutAuxValues(inputGradient + offset, auxData + offset,
                                     gradient + offset, alpha, blockSize);
    });

    return Status();
}


template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeBlockWithoutAuxValues(const algorithmFPType *inputGradient,
                                                                           const algorithmFPType *auxData,
                                                                                 algorithmFPType *gradient,
                                                                                 algorithmFPType alpha,
                                                                                 size_t blockSize)
{
    algorithmFPType *expValues = _intermediateValuesTls.local();
    BlockSizeType *indices = _indicesTls.local();

    BlockSizeType expValuesSize = 0;
    for (BlockSizeType i = 0; i < blockSize; i++)
    {
        if (auxData[i] < (algorithmFPType)0.0)
        {
            indices[expValuesSize] = i;
            expValues[expValuesSize] = auxData[i];
            expValuesSize++;
        }

        gradient[i] = inputGradient[i];
    }

    if (expValuesSize)
    {
        Math<algorithmFPType, cpu>::vExp(expValuesSize, expValues, expValues);
    }

  PRAGMA_VECTOR_ALWAYS
    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        expValues[i] *= alpha;
    }

  PRAGMA_IVDEP
    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        gradient[indices[i]] *= expValues[i];
    }
}


} // namespace internal
} // namespace backward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
