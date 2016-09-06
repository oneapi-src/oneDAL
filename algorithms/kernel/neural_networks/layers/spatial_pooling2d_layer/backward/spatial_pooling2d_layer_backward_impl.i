/* file: spatial_pooling2d_layer_backward_impl.i */
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
//  Implementation of backward pooling layer
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "maximum_pooling2d_layer_backward.h"
#include "stochastic_pooling2d_layer_backward.h"
#include "average_pooling2d_layer_backward.h"

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
void DAAL_EXPORT BasePoolingKernel<algorithmFPType, cpu>::compute(Tensor *inputGradientTensor, Tensor *gradientTensor, NumericTable *inputDimensions,
        Tensor *selectedPosTensor, const spatial_pooling2d::Parameter *parameter)
{
    SharedPtr<BasePoolingTask<algorithmFPType, cpu> >task(createTask(inputGradientTensor, gradientTensor, inputDimensions, selectedPosTensor, parameter));
    size_t L = parameter->pyramidHeight;
    size_t nSlices = inputGradientTensor->getDimensionSize(0);
    for(size_t slice = 0; slice < nSlices; slice++)
    {
        task->getSlice(slice);

        size_t accumulatedFlattenOffset = 0;
        size_t pow2 = 0;
        for(size_t level = 0, pow2 = 1; level < L; pow2 *= 2, level++)
        {
            task->preparePoolingTensors(level, accumulatedFlattenOffset);
            task->preparePoolingParameter(parameter, level);

            computePooling(task.get());

            mergeToResult(gradientTensor, task->onePoolingGradientTensor.get(), slice, task->targetOutLayout);

            accumulatedFlattenOffset += gradientTensor->getDimensionSize(6 - parameter->indices.size[0] - parameter->indices.size[1]) * pow2 * pow2;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
void DAAL_EXPORT BasePoolingKernel<algorithmFPType, cpu>::mergeToResult(Tensor *gradient, Tensor *partialGradient, size_t slice, const TensorOffsetLayout &targetOutLayout)
{
    ReadSubtensor<algorithmFPType, cpu, Tensor> partialGradientSubtensor(partialGradient, 0, 0, 0, 1);
    const algorithmFPType *partialGradientArray = partialGradientSubtensor.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> gradientSubtensor(gradient, 0, 0, slice, 1, targetOutLayout);
    algorithmFPType *gradientArray = gradientSubtensor.get();

    for(size_t i = 0; i < partialGradient->getSize(); i++)
    {
        gradientArray[i] += partialGradientArray[i];
    }
}

template<typename algorithmFPType, CpuType cpu>
void DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::maximum, cpu>::computePooling(BasePoolingTask<algorithmFPType, cpu> *taskBase)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::maximum, cpu> *task = (PoolingTask<algorithmFPType, spatial_pooling2d::internal::maximum, cpu> *)taskBase;

    size_t nInputDims = task->onePoolingGradientTensor->getDimensions().size();
    maximum_pooling2d::backward::Batch<algorithmFPType, maximum_pooling2d::defaultDense> backwardMaxPoolLayer(nInputDims);
    backwardMaxPoolLayer.input.set(layers::backward::inputGradient, task->poolingInputGradientTensor);
    backwardMaxPoolLayer.input.set(layers::backward::inputFromForward, services::SharedPtr<LayerData>(new LayerData()));
    backwardMaxPoolLayer.input.set(maximum_pooling2d::auxSelectedIndices, task->poolingSelectedPosTensor);
    backwardMaxPoolLayer.input.set(maximum_pooling2d::auxInputDimensions, task->poolingInputDimensions);
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&(task->pooling2dParameter), &backwardMaxPoolLayer.parameter);

    SharedPtr<maximum_pooling2d::backward::Result> maxPoolingResult(new maximum_pooling2d::backward::Result());

    maxPoolingResult->set(layers::backward::gradient, task->onePoolingGradientTensor);

    backwardMaxPoolLayer.setResult(maxPoolingResult);
    backwardMaxPoolLayer.compute();
}

template<typename algorithmFPType, CpuType cpu>
void DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::average, cpu>::computePooling(BasePoolingTask<algorithmFPType, cpu> *taskBase)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::average, cpu> *task = (PoolingTask<algorithmFPType, spatial_pooling2d::internal::average, cpu> *)taskBase;

    size_t nInputDims = task->onePoolingGradientTensor->getDimensions().size();
    average_pooling2d::backward::Batch<algorithmFPType, average_pooling2d::defaultDense> backwardAveragePoolLayer(nInputDims);
    backwardAveragePoolLayer.input.set(layers::backward::inputGradient, task->poolingInputGradientTensor);
    backwardAveragePoolLayer.input.set(layers::backward::inputFromForward, services::SharedPtr<LayerData>(new LayerData()));
    backwardAveragePoolLayer.input.set(average_pooling2d::auxInputDimensions, task->poolingInputDimensions);
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&(task->pooling2dParameter), &backwardAveragePoolLayer.parameter);

    SharedPtr<average_pooling2d::backward::Result> averagePoolingResult(new average_pooling2d::backward::Result());

    averagePoolingResult->set(layers::backward::gradient, task->onePoolingGradientTensor);

    backwardAveragePoolLayer.setResult(averagePoolingResult);
    backwardAveragePoolLayer.compute();
}

template<typename algorithmFPType, CpuType cpu>
void DAAL_EXPORT PoolingKernel<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu>::computePooling(BasePoolingTask<algorithmFPType, cpu> *taskBase)
{
    PoolingTask<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu> *task = (PoolingTask<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu> *)taskBase;

    size_t nInputDims = task->onePoolingGradientTensor->getDimensions().size();
    stochastic_pooling2d::backward::Batch<algorithmFPType, stochastic_pooling2d::defaultDense> backwardStochasticPoolLayer(nInputDims);
    backwardStochasticPoolLayer.input.set(layers::backward::inputGradient, task->poolingInputGradientTensor);
    backwardStochasticPoolLayer.input.set(layers::backward::inputFromForward, services::SharedPtr<LayerData>(new LayerData()));
    backwardStochasticPoolLayer.input.set(stochastic_pooling2d::auxSelectedIndices, task->poolingSelectedPosTensor);
    backwardStochasticPoolLayer.input.set(stochastic_pooling2d::auxInputDimensions, task->poolingInputDimensions);
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&(task->pooling2dParameter), &backwardStochasticPoolLayer.parameter);

    SharedPtr<stochastic_pooling2d::backward::Result> stochasticPoolingResult(new stochastic_pooling2d::backward::Result());

    stochasticPoolingResult->set(layers::backward::gradient, task->onePoolingGradientTensor);
    backwardStochasticPoolLayer.parameter.predictionStage = ((spatial_stochastic_pooling2d::Parameter *)(task->spatialParameter))->predictionStage;
    backwardStochasticPoolLayer.parameter.seed = ((spatial_stochastic_pooling2d::Parameter *)(task->spatialParameter))->seed;

    backwardStochasticPoolLayer.setResult(stochasticPoolingResult);
    backwardStochasticPoolLayer.compute();
}

} // namespace internal
} // namespace backward
} // namespace spatial_spatial_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
