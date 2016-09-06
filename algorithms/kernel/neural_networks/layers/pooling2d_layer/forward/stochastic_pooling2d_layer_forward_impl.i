/* file: stochastic_pooling2d_layer_forward_impl.i */
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

#ifndef __STOCHASTIC_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __STOCHASTIC_POOLING2D_LAYER_FORWARD_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace stochastic_pooling2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(Tensor *dataTensor, Tensor *valueTensor,
        Tensor *selectedPosTensor, const Parameter *parameter)
{
    size_t nDims = dataTensor->getNumberOfDimensions();
    const Collection<size_t> &dims = dataTensor->getDimensions();
    const Collection<size_t> &valueDims = valueTensor->getDimensions();

    pooling2d::internal::Parameter par(parameter->indices.size, parameter->paddings.size,
                                       parameter->strides.size, parameter->kernelSizes.size,
                                       dataTensor, dims, valueDims);

    Collection<size_t> extractLayoutCollection(nDims);
    for(size_t i = 0; i < nDims; i++)
    {
        extractLayoutCollection[i] = i;
    }

    extractLayoutCollection[par.firstIndex] = nDims - 2;
    extractLayoutCollection[par.secondIndex] = nDims - 1;
    extractLayoutCollection[nDims - 2] = par.firstIndex;
    extractLayoutCollection[nDims - 1] = par.secondIndex;

    TensorOffsetLayout targetInLayout = dataTensor->createDefaultSubtensorLayout();
    targetInLayout.shuffleDimensions(extractLayoutCollection);

    TensorOffsetLayout targetOutLayout = valueTensor->createDefaultSubtensorLayout();
    targetOutLayout.shuffleDimensions(extractLayoutCollection);

    ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor(dataTensor, 0, 0, 0, dims[0], targetInLayout);
    WriteSubtensor<algorithmFPType, cpu, Tensor> valueSubtensor(valueTensor, 0, 0, 0, valueDims[0], targetOutLayout);
    WriteSubtensor<int, cpu, Tensor> selectedPosSubtensor;

    int *selectedPos = nullptr;
    size_t selectedPosSize = 0;

    bool trainingStage = !(parameter->predictionStage);
    if(trainingStage)
    {
        TensorOffsetLayout targetPosLayout = selectedPosTensor->createDefaultSubtensorLayout();
        targetPosLayout.shuffleDimensions(extractLayoutCollection);
        selectedPosSubtensor.set(*selectedPosTensor, 0, 0, 0, valueDims[0], targetPosLayout);
        selectedPos = selectedPosSubtensor.get();
        DAAL_CHECK(selectedPos, ErrorMemoryAllocationFailed);
        selectedPosSize = selectedPosTensor->getSize();
        invIntMaxVal = 1.0 / data_feature_utils::internal::MaxVal<int, cpu>::get();
    }

    size_t nSlices = dataTensor->getSize() / (dims[par.firstIndex] * dims[par.secondIndex]);
    size_t dataSliceSize = dims[par.firstIndex] * dims[par.secondIndex];
    size_t valueSliceSize = valueDims[par.firstIndex] * valueDims[par.secondIndex];
    size_t nFlatten = par.firstKernelSize * par.secondKernelSize;

    TSmartPtr<algorithmFPType, cpu> kernelWeightsPtr(nFlatten);
    algorithmFPType *kernelWeights = kernelWeightsPtr.get();
    DAAL_CHECK(kernelWeights, ErrorMemoryAllocationFailed);

    if(selectedPosSize != 0)
    {
        getUniformRandFrom0to1(selectedPos, selectedPosSize, parameter->seed);
    }

    const algorithmFPType *data = dataSubtensor.get();
    DAAL_CHECK(data, ErrorMemoryAllocationFailed);

    algorithmFPType *value = valueSubtensor.get();
    DAAL_CHECK(value, ErrorMemoryAllocationFailed);

    for (MKL_INT i = 0; i < nSlices; i++)
    {
        const algorithmFPType *dataSlice = &data[i * dataSliceSize];
        algorithmFPType *valueSlice = &value[i * valueSliceSize];
        int *selectedPosSlice = nullptr;
        algorithmFPType *uniformRand = nullptr;
        if(trainingStage)
        {
            selectedPosSlice = &selectedPos[i * valueSliceSize];
        }
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (MKL_INT f = -par.firstPadding, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            /*
            * Loop by the second kernel dimension
            * s - index of the left upper corner of the kernel
            * so - index of the output value
            */
            for (MKL_INT s = -par.secondPadding, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
            {
                for(size_t l = 0; l < nFlatten; l++)
                {
                    kernelWeights[l] = (algorithmFPType)0.0;
                }

                algorithmFPType sum = (algorithmFPType)0.0;
                /*
                 * Loops over the kernel to get weights
                 */
                for (MKL_INT fi = f; fi < f + par.firstKernelSize; fi++)
                {
                    for (MKL_INT si = s; si < s + par.secondKernelSize; si++)
                    {
                        MKL_INT dataIndex = si + par.secondSize * fi;
                        algorithmFPType dataValue = (par.getPaddingFlag(fi, si) ? 0.0 : dataSlice[dataIndex]);

                        sum += dataValue;
                        size_t l = (fi - f) * par.secondKernelSize + (si - s);
                        kernelWeights[l] = dataValue;
                    }
                }

                algorithmFPType invSum = 1.0 / sum;
                for(size_t l = 0; l < nFlatten; l++)
                {
                    kernelWeights[l] *= invSum;
                }

                MKL_INT valueIndex = so + par.secondOutSize * fo;

                if(trainingStage)
                {
                    getMultivariateRandomDataValue(dataSlice, f, s, kernelWeights, nFlatten, par, valueSlice[valueIndex], selectedPosSlice[valueIndex]);
                }
                else
                {
                    computeWeightedAverage(dataSlice, f, s, kernelWeights, par, valueSlice[valueIndex]);
                }
            }
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::computeWeightedAverage(
    const algorithmFPType *dataSlice,
    MKL_INT f,
    MKL_INT s,
    algorithmFPType *kernelWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value)
{
    value = 0;
    for (MKL_INT fi = f; fi < f + par.firstKernelSize; fi++)
    {
        for (MKL_INT si = s; si < s + par.secondKernelSize; si++)
        {
            MKL_INT dataIndex = si + par.secondSize * fi;
            algorithmFPType dataValue = (par.getPaddingFlag(fi, si) ? 0.0 : dataSlice[dataIndex]);

            size_t l = (fi - f) * par.secondKernelSize + (si - s);
            value += kernelWeights[l] * dataValue;
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::getMultivariateRandomDataValue(
    const algorithmFPType *dataSlice,
    MKL_INT f,
    MKL_INT s,
    algorithmFPType *weights,
    size_t nWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value,
    int &selectedPos)
{
    selectedPos = getMultinomialRandomValue(weights, nWeights, selectedPos);
    size_t fi = f + (selectedPos / par.firstKernelSize);
    size_t si = s + (selectedPos - (fi - f) * par.firstKernelSize);

    MKL_INT dataIndex = si + par.secondSize * fi;
    algorithmFPType dataValue = (par.getPaddingFlag(fi, si) ? 0.0 : dataSlice[dataIndex]);
    value = dataValue;
}

template<typename algorithmFPType, Method method, CpuType cpu>
size_t PoolingKernel<algorithmFPType, method, cpu>::getMultinomialRandomValue(algorithmFPType *weights, size_t nWeights, const int uniformRandVal)
{
    algorithmFPType randFrom0To1 = uniformRandVal * invIntMaxVal;
    algorithmFPType sum = 0;
    size_t returnVal = 0;
    while(sum <= randFrom0To1 && returnVal < nWeights)
    {
        sum += weights[returnVal];
        returnVal++;
    }
    return returnVal - 1;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::getUniformRandFrom0to1(int *uniformRand, size_t nUniformRand, size_t seed)
{
    daal::internal::IntRng <int, cpu> rng((int)seed);
    int intMaxVal = data_feature_utils::internal::MaxVal<int, cpu>::get();
    rng.uniform(nUniformRand, 0, intMaxVal, uniformRand);
}

} // namespace internal
} // namespace forward
} // namespace stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
