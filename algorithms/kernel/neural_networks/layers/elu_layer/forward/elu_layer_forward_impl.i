/* file: elu_layer_forward_impl.i */
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
//  Implementation of forward ELU layer
//--
*/

#ifndef __ELU_LAYER_FORWARD_IMPL_I__
#define __ELU_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

using namespace daal::internal;

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::compute(const Parameter &parameter,
                                                        const Tensor &dataTensor,
                                                              Tensor &valueTensor,
                                                              Tensor *auxValueTensor)
{
    const algorithmFPType alpha = parameter.alpha;
    const bool predictionStage  = parameter.predictionStage;

    /* if auxValues == nullptr we assume that layer is on prediction stage */
    Tensor *auxValues = (predictionStage) ? nullptr : auxValueTensor;

    if (elu::internal::canComputeInMklLayout<algorithmFPType, cpu>(dataTensor, valueTensor))
    {
        return computeInMKLLayout(dataTensor, valueTensor, auxValues, alpha);
    }
    else
    {
        return computeLayoutAgnostic(dataTensor, valueTensor, auxValues, alpha);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::computeLayoutAgnostic(const Tensor &dataTensor,
                                                                            Tensor &valueTensor,
                                                                            Tensor *auxValueTensor,
                                                                            algorithmFPType alpha)
{
    ReadSubtensor<algorithmFPType, cpu> dataBlock(const_cast<Tensor &>(dataTensor));
    DAAL_CHECK_BLOCK_STATUS(dataBlock);

    WriteSubtensor<algorithmFPType, cpu> valueBlock(valueTensor);
    DAAL_CHECK_BLOCK_STATUS(valueBlock);

    if (auxValueTensor)
    {
        WriteSubtensor<algorithmFPType, cpu> auxValueBlock(auxValueTensor);
        DAAL_CHECK_BLOCK_STATUS(auxValueBlock);

        computeInRawLayout(dataBlock.get(), valueBlock.get(), auxValueBlock.get(),
                           alpha, dataTensor.getSize());
    }
    else
    {
        computeInRawLayoutPrediction(dataBlock.get(), valueBlock.get(),
                                     alpha, dataTensor.getSize());
    }

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status ELUKernel<algorithmFPType, method, cpu>::computeInMKLLayout(const Tensor &dataTensor,
                                                                         Tensor &valueTensor,
                                                                         Tensor *auxValueTensor,
                                                                         algorithmFPType alpha)
{
    using MklTensorType = MklTensor<algorithmFPType>;

    /* We assume tensros can be casted and check was performed by the caller of this function */
    auto &dataMklTensor  = const_cast<MklTensorType &>(static_cast<const MklTensorType &>(dataTensor));
    auto &valueMklTensor = static_cast<MklTensorType &>(valueTensor);
    valueMklTensor.setDnnLayout(dataMklTensor.getSharedDnnLayout());

    const algorithmFPType *data = dataMklTensor.getDnnArray();
    algorithmFPType *value = valueMklTensor.getDnnArray();

    if (auxValueTensor)
    {
        WriteSubtensor<algorithmFPType, cpu> auxValueBlock(auxValueTensor);
        DAAL_CHECK_BLOCK_STATUS(auxValueBlock);

        computeInRawLayout(data, value, auxValueBlock.get(), alpha, dataTensor.getSize());
    }
    else
    {
        computeInRawLayoutPrediction(data, value, alpha, dataTensor.getSize());
    }

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeInRawLayout(const algorithmFPType *data,
                                                                       algorithmFPType *value,
                                                                       algorithmFPType *auxValue,
                                                                       algorithmFPType alpha,
                                                                       size_t dataSize)
{
    elu::internal::computeThreaded<algorithmFPType, cpu>(dataSize,
    [ & ](size_t offset, size_t blockSize)
    {
        computeBlock(data + offset, value + offset, auxValue + offset, alpha, blockSize);
    });
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeInRawLayoutPrediction(const algorithmFPType *data,
                                                                                 algorithmFPType *value,
                                                                                 algorithmFPType alpha,
                                                                                 size_t dataSize)
{
    elu::internal::computeThreaded<algorithmFPType, cpu>(dataSize,
    [ & ](size_t offset, size_t blockSize)
    {
        computeBlockPrediction(data + offset, value + offset, alpha, blockSize);
    });
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeBlock(const algorithmFPType *data,
                                                                 algorithmFPType *value,
                                                                 algorithmFPType *auxValue,
                                                                 algorithmFPType alpha,
                                                                 size_t blockSize)
{
    BlockSizeType expValuesSize = 0;
    BlockSizeType *indices = _indicesTls.local();

    for (BlockSizeType i = 0; i < blockSize; i++)
    {
        if (data[i] < (algorithmFPType)0.0)
        {
            indices[expValuesSize] = i;
            auxValue[expValuesSize++] = data[i];
        }

        value[i] = data[i];
    }

    if (expValuesSize)
    {
        Math<algorithmFPType, cpu>::vExp(expValuesSize, auxValue, auxValue);
    }

  PRAGMA_VECTOR_ALWAYS
    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        auxValue[i] *= alpha;
    }

  PRAGMA_IVDEP
    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        value[indices[i]] = auxValue[i] - alpha;
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void ELUKernel<algorithmFPType, method, cpu>::computeBlockPrediction(const algorithmFPType *data,
                                                                           algorithmFPType *value,
                                                                           algorithmFPType alpha,
                                                                           size_t blockSize)
{
    algorithmFPType *expValues = _intermediateValuesTls.local();
    BlockSizeType *indices = _indicesTls.local();

    BlockSizeType expValuesSize = 0;
    for (BlockSizeType i = 0; i < blockSize; i++)
    {
        if (data[i] < (algorithmFPType)0.0)
        {
            indices[expValuesSize] = i;
            expValues[expValuesSize++] = data[i];
        }

        value[i] = data[i];
    }

    if (expValuesSize)
    {
        Math<algorithmFPType, cpu>::vExp(expValuesSize, expValues, expValues);
    }

  PRAGMA_VECTOR_ALWAYS
    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        expValues[i] = expValues[i] * alpha - alpha;
    }

    for (BlockSizeType i = 0; i < expValuesSize; i++)
    {
        value[indices[i]] = expValues[i];
    }
}

} // namespace internal
} // namespace forward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
