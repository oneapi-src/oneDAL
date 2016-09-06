/* file: spatial_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward pooling layer
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __SPATIAL_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

#include "tensor.h"
#include "homogen_numeric_table.h"
#include "maximum_pooling2d_layer_forward.h"
#include "stochastic_pooling2d_layer_forward.h"
#include "average_pooling2d_layer_forward.h"

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
void BasePoolingKernel<algorithmFPType, cpu>::compute(Tensor *dataTensor, Tensor *valueTensor,
        Tensor *selectedPosTensor, const spatial_pooling2d::Parameter *spatialParameter)
{
    size_t L = spatialParameter->pyramidHeight;

    size_t nDims = 4;
    Collection<size_t> extractLayoutCollection(nDims);
    for(size_t i = 0; i < nDims; i++)
    {
        extractLayoutCollection[i] = i;
    }

    swap<size_t, cpu>(extractLayoutCollection[spatialParameter->indices.size[0]], extractLayoutCollection[nDims - 2]);
    swap<size_t, cpu>(extractLayoutCollection[spatialParameter->indices.size[1]], extractLayoutCollection[nDims - 1]);

    TensorOffsetLayout targetInLayout = dataTensor->createDefaultSubtensorLayout();
    targetInLayout.shuffleDimensions(extractLayoutCollection);

    Collection<size_t> dims(targetInLayout.getDimensions());
    const Collection<size_t> &valueDims = valueTensor->getDimensions();

    ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor;
    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> valueSubtensor;
    WriteOnlySubtensor<int, cpu, Tensor> selectedPosSubtensor;

    size_t poolDimSize1 = dims[2];
    size_t poolDimSize2 = dims[3];

    Collection<size_t> dataSliceDims(dims);
    dataSliceDims[0] = 1;

    size_t flattenSize = valueDims[1];
    pooling2d::forward::Result dummyResult;
    for(size_t slice = 0; slice < dims[0]; slice++)
    {
        dataSubtensor.set(*dataTensor, 0, 0, slice, 1, targetInLayout);
        const algorithmFPType *dataSlice = dataSubtensor.get();
        TensorPtr dataSliceTensor(new HomogenTensor<algorithmFPType>(dataSliceDims, const_cast<algorithmFPType *>(dataSlice)));

        valueSubtensor.set(*valueTensor, 0, 0, slice, 1);
        algorithmFPType *valueSlice = valueSubtensor.get();

        int *selectedPosSlice = nullptr;
        if(selectedPosTensor)
        {
            selectedPosSubtensor.set(*selectedPosTensor, 0, 0, slice, 1);
            selectedPosSlice = selectedPosSubtensor.get();
        }

        size_t accumulatedFlattenOffset = 0;
        for(size_t level = 0, pow2 = 1; level < L ; level++, pow2 *= 2)
        {
            size_t kernelSize1 = (poolDimSize1 % pow2 == 0) ? poolDimSize1 / pow2 : poolDimSize1 / pow2 + 1;
            size_t kernelSize2 = (poolDimSize2 % pow2 == 0) ? poolDimSize2 / pow2 : poolDimSize2 / pow2 + 1;
            size_t stride1 = kernelSize1;
            size_t stride2 = kernelSize2;
            size_t padding1 = (kernelSize1 * pow2 - poolDimSize1 + 1) / 2;
            size_t padding2 = (kernelSize2 * pow2 - poolDimSize2 + 1) / 2;


            pooling2d::Parameter poolingPar(2, 3, kernelSize1, kernelSize2, stride1, stride2, padding1, padding2);

            algorithmFPType *value = &valueSlice[accumulatedFlattenOffset];

            Collection<size_t> valueDims = dummyResult.getValueSize(dataSliceTensor->getDimensions(), &poolingPar, 0);
            TensorPtr poolingValueTensor(new HomogenTensor<algorithmFPType>(valueDims, value));

            TensorPtr poolingSelectedPosTensor;
            if(selectedPosTensor)
            {
                int *selectedPos = &selectedPosSlice[accumulatedFlattenOffset];
                poolingSelectedPosTensor = TensorPtr(new HomogenTensor<int>(valueDims, selectedPos));
            }

            computePooling(poolingPar, spatialParameter, dataSliceTensor, poolingValueTensor, poolingSelectedPosTensor);
            accumulatedFlattenOffset += dims[1] * pow2 * pow2;
        }
    }
}

template<typename algorithmFPType, spatial_pooling2d::internal::Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::computePooling(
    pooling2d::Parameter poolingPar,
    const spatial_pooling2d::Parameter *spatialParameter,
    TensorPtr dataTensor,
    TensorPtr poolingValueTensor,
    TensorPtr poolingSelectedPosTensor)
{}


template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, spatial_pooling2d::internal::maximum, cpu>::computePooling(
    pooling2d::Parameter poolingPar,
    const spatial_pooling2d::Parameter *spatialParameter,
    TensorPtr dataTensor,
    TensorPtr poolingValueTensor,
    TensorPtr poolingSelectedPosTensor)
{
    size_t nInputDims = dataTensor->getDimensions().size();
    maximum_pooling2d::forward::Batch<algorithmFPType, maximum_pooling2d::defaultDense> forwardMaxPoolLayer(nInputDims);
    forwardMaxPoolLayer.input.set(layers::forward::data, dataTensor);
    forwardMaxPoolLayer.parameter.predictionStage = spatialParameter->predictionStage;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&poolingPar, &forwardMaxPoolLayer.parameter);

    SharedPtr<maximum_pooling2d::forward::Result> maxPoolingResult(new maximum_pooling2d::forward::Result());

    maxPoolingResult->set(layers::forward::value, poolingValueTensor);
    maxPoolingResult->set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
    maxPoolingResult->set(maximum_pooling2d::auxSelectedIndices, poolingSelectedPosTensor);
    maxPoolingResult->set(maximum_pooling2d::auxInputDimensions, SharedPtr<NumericTable>(new HomogenNumericTable<int>(nInputDims, 1, NumericTable::doAllocate)));

    forwardMaxPoolLayer.setResult(maxPoolingResult);
    forwardMaxPoolLayer.compute();
}

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu>::computePooling(
    pooling2d::Parameter poolingPar,
    const spatial_pooling2d::Parameter *spatialParameter,
    TensorPtr dataTensor,
    TensorPtr poolingValueTensor,
    TensorPtr poolingSelectedPosTensor)
{
    size_t nInputDims = dataTensor->getDimensions().size();
    stochastic_pooling2d::forward::Batch<algorithmFPType, stochastic_pooling2d::defaultDense> forwardStochasticPoolLayer(nInputDims);
    forwardStochasticPoolLayer.input.set(layers::forward::data, dataTensor);
    forwardStochasticPoolLayer.parameter.predictionStage = spatialParameter->predictionStage;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&poolingPar, &forwardStochasticPoolLayer.parameter);

    SharedPtr<stochastic_pooling2d::forward::Result> stochasticPoolingResult(new stochastic_pooling2d::forward::Result());

    stochasticPoolingResult->set(layers::forward::value, poolingValueTensor);
    stochasticPoolingResult->set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
    stochasticPoolingResult->set(stochastic_pooling2d::auxSelectedIndices, poolingSelectedPosTensor);
    stochasticPoolingResult->set(stochastic_pooling2d::auxInputDimensions, SharedPtr<NumericTable>(new HomogenNumericTable<int>(nInputDims, 1, NumericTable::doAllocate)));

    forwardStochasticPoolLayer.setResult(stochasticPoolingResult);
    forwardStochasticPoolLayer.parameter.predictionStage = ((const spatial_stochastic_pooling2d::Parameter *)spatialParameter)->predictionStage;
    forwardStochasticPoolLayer.parameter.seed = ((const spatial_stochastic_pooling2d::Parameter *)spatialParameter)->seed;
    forwardStochasticPoolLayer.compute();
}

template<typename algorithmFPType, CpuType cpu>
void PoolingKernel<algorithmFPType, spatial_pooling2d::internal::average, cpu>::computePooling(
    pooling2d::Parameter poolingPar,
    const spatial_pooling2d::Parameter *spatialParameter,
    TensorPtr dataTensor,
    TensorPtr poolingValueTensor,
    TensorPtr poolingSelectedPosTensor)
{
    size_t nInputDims = dataTensor->getDimensions().size();
    average_pooling2d::forward::Batch<algorithmFPType, average_pooling2d::defaultDense> forwardAveragePoolLayer(nInputDims);
    forwardAveragePoolLayer.input.set(layers::forward::data, dataTensor);
    forwardAveragePoolLayer.parameter.predictionStage = spatialParameter->predictionStage;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(&poolingPar, &forwardAveragePoolLayer.parameter);

    SharedPtr<average_pooling2d::forward::Result> averagePoolingResult(new average_pooling2d::forward::Result());

    averagePoolingResult->set(layers::forward::value, poolingValueTensor);
    averagePoolingResult->set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
    averagePoolingResult->set(average_pooling2d::auxInputDimensions, SharedPtr<NumericTable>(new HomogenNumericTable<int>(nInputDims, 1, NumericTable::doAllocate)));

    forwardAveragePoolLayer.setResult(averagePoolingResult);
    forwardAveragePoolLayer.compute();
}

} // namespace internal
} // namespace forward
} // namespace spatial_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
