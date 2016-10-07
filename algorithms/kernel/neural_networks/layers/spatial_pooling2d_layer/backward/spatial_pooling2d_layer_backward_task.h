/* file: spatial_pooling2d_layer_backward_task.h */
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

//++
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_TASK_H__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_TASK_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_backward_types.h"
#include "spatial_pooling2d_layer_internal_types.h"
#include "tensor.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
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
class DAAL_EXPORT BasePoolingTask
{
public:
    BasePoolingTask(Tensor *_inputGradientTensor,
                    Tensor *_gradientTensor,
                    NumericTable *_inputDimensions,
                    const spatial_pooling2d::Parameter *_parameter) :
        inputGradientTensor(_inputGradientTensor),
        spatialParameter(_parameter),
        pooling2dParameter(0, 0, 0, 0, 0, 0, 0, 0),
        onePoolingGradientArray(0),
        targetOutLayout(_gradientTensor->createDefaultSubtensorLayout())
    {
        size_t nDims = _gradientTensor->getNumberOfDimensions();
        Collection<size_t> extractLayoutCollection(nDims);
        for(size_t i = 0; i < nDims; i++)
        {
            extractLayoutCollection[i] = i;
        }
        swap<size_t, cpu>(extractLayoutCollection[spatialParameter->indices.size[0]], extractLayoutCollection[nDims - 2]);
        swap<size_t, cpu>(extractLayoutCollection[spatialParameter->indices.size[1]], extractLayoutCollection[nDims - 1]);

        targetOutLayout.shuffleDimensions(extractLayoutCollection);
        oneGradientDims = targetOutLayout.getDimensions();
        oneGradientDims[0] = 1;

        onePoolingGradientArray.reset(_gradientTensor->getSize() / _gradientTensor->getDimensionSize(0));

        poolingInputDimensions = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(nDims, 1 , NumericTable::doAllocate));
        WriteOnlyRows<algorithmFPType, cpu> poolingInputDimensionsBD(*poolingInputDimensions, 0, 1);
        algorithmFPType *inputDimsSlice = poolingInputDimensionsBD.get();
        for(size_t i = 0; i < nDims; i++)
        {
            inputDimsSlice[i] = oneGradientDims[i];
        }

        WriteOnlySubtensor<algorithmFPType, cpu, Tensor> gradientSubtensor(_gradientTensor, 0, 0, 0, _gradientTensor->getDimensionSize(0));
        algorithmFPType *gradientArray = gradientSubtensor.get();
        for(size_t i = 0; i < _gradientTensor->getSize(); i++)
        {
            gradientArray[i] = 0;
        }
    }

    virtual void getSlice(size_t slice)
    {
        getInputGradientSlice(slice);
        getSelectedPosSlice(slice);
    }

    virtual void getInputGradientSlice(size_t slice)
    {
        inputGradientSubtensor.set(*inputGradientTensor, 0, 0, slice, 1);
    }

    virtual void getSelectedPosSlice(size_t slice) {}

    void prepareCommonPoolingTensors(size_t level, size_t offset)
    {
        inputGradientSliceDims = oneGradientDims;
        inputGradientSliceDims[2] = ((size_t)1 << level);
        inputGradientSliceDims[3] = ((size_t)1 << level);

        const algorithmFPType *inputGradientData = inputGradientSubtensor.get() + offset;
        poolingInputGradientTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(inputGradientSliceDims, const_cast<algorithmFPType *>(inputGradientData)));

        onePoolingGradientTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(oneGradientDims, onePoolingGradientArray.get()));

    }

    virtual void preparePoolingTensors(size_t level, size_t offset)
    {
        prepareCommonPoolingTensors(level, offset);
    }

    void preparePoolingParameter(const spatial_pooling2d::Parameter *parameter, size_t level)
    {
        size_t pow2 = (size_t)1 << level;
        const Collection<size_t> &oneGradientDims = onePoolingGradientTensor->getDimensions();
        pooling2dParameter.indices.size[0] = 2;
        pooling2dParameter.indices.size[1] = 3;
        pooling2dParameter.kernelSizes.size[0] = (oneGradientDims[2] % pow2 == 0) ? oneGradientDims[2] / pow2 : oneGradientDims[2] / pow2 + 1;
        pooling2dParameter.kernelSizes.size[1] = (oneGradientDims[3] % pow2 == 0) ? oneGradientDims[3] / pow2 : oneGradientDims[3] / pow2 + 1;
        pooling2dParameter.strides.size[0] = pooling2dParameter.kernelSizes.size[0];
        pooling2dParameter.strides.size[1] = pooling2dParameter.kernelSizes.size[1];
        pooling2dParameter.paddings.size[0] = (pooling2dParameter.kernelSizes.size[0] * pow2 - oneGradientDims[2] + 1) / 2;
        pooling2dParameter.paddings.size[1] = (pooling2dParameter.kernelSizes.size[1] * pow2 - oneGradientDims[3] + 1) / 2;
    }

    virtual ~BasePoolingTask() {}

    Tensor *inputGradientTensor;
    ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientSubtensor;

    pooling2d::Parameter pooling2dParameter;

    const spatial_pooling2d::Parameter *spatialParameter;
    Collection<size_t> oneGradientDims;
    Collection<size_t> inputGradientSliceDims;

    TensorOffsetLayout targetOutLayout;

    TSmartPtr<algorithmFPType, cpu> onePoolingGradientArray;

    SharedPtr<Tensor> poolingInputGradientTensor;
    SharedPtr<Tensor> onePoolingGradientTensor;
    SharedPtr<NumericTable> poolingInputDimensions;
};

template<typename algorithmFPType, spatial_pooling2d::internal::Method method, CpuType cpu>
class DAAL_EXPORT PoolingTask : public BasePoolingTask<algorithmFPType, cpu>
{
public:
    using BasePoolingTask<algorithmFPType, cpu>::getInputGradientSlice;
    using BasePoolingTask<algorithmFPType, cpu>::inputGradientSliceDims;
    using BasePoolingTask<algorithmFPType, cpu>::prepareCommonPoolingTensors;

    PoolingTask(
        Tensor *_inputGradientTensor,
        Tensor *_gradientTensor,
        NumericTable *_inputDimensions,
        Tensor *_selectedPosTensor,
        const spatial_pooling2d::Parameter *parameter) :
        BasePoolingTask<algorithmFPType, cpu>(_inputGradientTensor, _gradientTensor, _inputDimensions, parameter), selectedPosTensor(_selectedPosTensor)
    {}

    void getSelectedPosSlice(size_t slice) DAAL_C11_OVERRIDE
    {
        selectedPosSubtensor.set(*selectedPosTensor, 0, 0, slice, 1);
    }

    void getSlice(size_t slice) DAAL_C11_OVERRIDE
    {
        getInputGradientSlice(slice);
        getSelectedPosSlice(slice);
    }

    void preparePoolingTensors(size_t level, size_t offset) DAAL_C11_OVERRIDE
    {
        prepareCommonPoolingTensors(level, offset);

        algorithmFPType *selectedPosData = selectedPosSubtensor.get() + offset;
        poolingSelectedPosTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(inputGradientSliceDims, selectedPosData));
    }

    Tensor *selectedPosTensor;
    WriteSubtensor<algorithmFPType, cpu, Tensor> selectedPosSubtensor;

    SharedPtr<Tensor> poolingSelectedPosTensor;
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingTask<algorithmFPType, spatial_pooling2d::internal::average, cpu> : public BasePoolingTask<algorithmFPType, cpu>
{
public:
    PoolingTask(Tensor *_inputGradientTensor, Tensor *_gradientTensor, NumericTable *_inputDimensions, const spatial_pooling2d::Parameter *parameter) :
        BasePoolingTask<algorithmFPType, cpu>(_inputGradientTensor, _gradientTensor, _inputDimensions, parameter)
    {}
};

} // internal
} // backward
} // spatial_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
