/* file: spatial_pooling2d_layer_backward_impl.i */
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
//  Implementation of backward pooling layer
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "stochastic_pooling2d_layer_backward.h"
#include "average_pooling2d_layer_backward.h"
#include "spatial_pooling2d_layer_backward_task.h"

#include "maximum_pooling2d_layer_backward_kernel.h"
#include "average_pooling2d_layer_backward_kernel.h"

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
namespace spatial_pooling2d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
Status DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::maximum, cpu>::compute(
    const Tensor &inputGradientTensor, Tensor &gradientTensor, const Tensor &selectedPosTensor, const spatial_maximum_pooling2d::Parameter &parameter)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::maximum, cpu> task(inputGradientTensor, gradientTensor, selectedPosTensor, parameter);
    return task.proceed();
}

template<typename algorithmFPType, CpuType cpu>
Status DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::average, cpu>::compute(
    const Tensor &inputGradientTensor, Tensor &gradientTensor, const spatial_average_pooling2d::Parameter &parameter)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::average, cpu> task(inputGradientTensor, gradientTensor, parameter);
    return task.proceed();
}

template<typename algorithmFPType, CpuType cpu>
Status DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu>::compute(
    const Tensor &inputGradientTensor, Tensor &gradientTensor, const Tensor &selectedPosTensor, const spatial_stochastic_pooling2d::Parameter &parameter)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu> task(inputGradientTensor, gradientTensor, selectedPosTensor, parameter);
    return task.proceed();
}

template<typename algorithmFPType, CpuType cpu>
services::Status DAAL_EXPORT BasePoolingTask<algorithmFPType, cpu>::proceed()
{
    Status s;
    DAAL_CHECK_STATUS(s, init());

    size_t L = spatialParameter.pyramidHeight;
    size_t nSlices = inputGradientTensor.getDimensionSize(0);
    for(size_t slice = 0; slice < nSlices; slice++)
    {
        DAAL_CHECK_STATUS(s, getSlice(slice));

        size_t accumulatedFlattenOffset = 0;
        size_t pow2 = 0;
        for(size_t level = 0, pow2 = 1; level < L; pow2 *= 2, level++)
        {
            DAAL_CHECK_STATUS(s, preparePoolingTensors(level, accumulatedFlattenOffset));
            DAAL_CHECK_STATUS(s, preparePoolingParameter(level));

            DAAL_CHECK_STATUS(s, computePooling());

            DAAL_CHECK_STATUS(s, mergeToResult(slice));

            accumulatedFlattenOffset += gradientTensor.getDimensionSize(6 - spatialParameter.indices.size[0] - spatialParameter.indices.size[1]) * pow2 * pow2;
        }
    }
    return s;
}

template<typename algorithmFPType, CpuType cpu>
Status DAAL_EXPORT BasePoolingTask<algorithmFPType, cpu>::mergeToResult(const size_t slice)
{
    ReadSubtensor<algorithmFPType, cpu, Tensor> partialGradientSubtensor(poolingGradientTensor.get(), 0, 0, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(partialGradientSubtensor);
    const algorithmFPType *partialGradientArray = partialGradientSubtensor.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> gradientSubtensor(gradientTensor, 0, 0, slice, 1, targetOutLayout);
    DAAL_CHECK_BLOCK_STATUS(gradientSubtensor);
    algorithmFPType *gradientArray = gradientSubtensor.get();

    const size_t blockSize = 2048;
    const size_t n = poolingGradientTensor->getSize();
    size_t nBlocks = n / blockSize;
    if (nBlocks * blockSize < n) { nBlocks++; }

    threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t endBlock = (iBlock + 1) * blockSize > n ? n : (iBlock + 1) * blockSize;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = iBlock * blockSize; i < endBlock ; i++)
        {
            gradientArray[i] += partialGradientArray[i];
        }
    } );

    return Status();
}


template<typename algorithmFPType, spatial_pooling2d::internal::Method method, CpuType cpu>
Status DAAL_EXPORT PoolingTask<algorithmFPType, method, cpu>::computePooling()
{
    const services::Collection<size_t>& inDimsFull  = this->poolingInputGradientTensor->getDimensions();
    const services::Collection<size_t>& outDimsFull = this->poolingGradientTensor->getDimensions();

    using MaxPoolingBackwardKernel = layers::maximum_pooling2d::backward::internal::PoolingKernel<
        algorithmFPType, layers::maximum_pooling2d::defaultDense, cpu>;

    Status status;
    MaxPoolingBackwardKernel maxPoolKernel;
    DAAL_CHECK_STATUS(status, maxPoolKernel.initialize(inDimsFull, outDimsFull));

    Tensor *dataTensor = nullptr;
    status |= maxPoolKernel.compute(*(this->poolingInputGradientTensor),
                                    *(this->poolingSelectedPosTensor),
                                    *(this->poolingGradientTensor),
                                     dataTensor, this->poolingParameter);
    return status;
}

template<typename algorithmFPType, CpuType cpu>
Status DAAL_EXPORT PoolingTask<algorithmFPType, spatial_pooling2d::internal::average, cpu>::computePooling()
{
    const services::Collection<size_t>& inDimsFull  = this->poolingInputGradientTensor->getDimensions();
    const services::Collection<size_t>& outDimsFull = this->poolingGradientTensor->getDimensions();

    using AvePoolingBackwardKernel = layers::average_pooling2d::backward::internal::PoolingKernel<
        algorithmFPType, layers::average_pooling2d::defaultDense, cpu>;

    Status status;
    AvePoolingBackwardKernel avePoolKernel;
    DAAL_CHECK_STATUS(status, avePoolKernel.initialize(inDimsFull, outDimsFull));

    Tensor *dataTensor = nullptr;
    status |= avePoolKernel.compute(*(this->poolingInputGradientTensor),
                                      this->poolingParameter,
                                    *(this->poolingGradientTensor),
                                     dataTensor);
    return status;
}

} // namespace internal
} // namespace backward
} // namespace spatial_spatial_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
