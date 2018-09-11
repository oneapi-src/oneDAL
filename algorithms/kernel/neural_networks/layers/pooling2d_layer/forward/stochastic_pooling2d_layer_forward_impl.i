/* file: stochastic_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward pooling layer
//--
*/

#ifndef __STOCHASTIC_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __STOCHASTIC_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "threading.h"
#include "uniform_kernel.h"
#include "uniform_impl.i"

using namespace daal::algorithms::distributions::uniform::internal;

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
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &dataTensor, Tensor &valueTensor,
        Tensor *selectedPosTensor, const Parameter &parameter, engines::BatchBase &engine)
{
    size_t nDims = dataTensor.getNumberOfDimensions();
    const Collection<size_t> &dims = dataTensor.getDimensions();
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    pooling2d::internal::Parameter par(parameter.indices.size, parameter.paddings.size,
                                       parameter.strides.size, parameter.kernelSizes.size,
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

    TensorOffsetLayout targetInLayout = dataTensor.createDefaultSubtensorLayout();
    targetInLayout.shuffleDimensions(extractLayoutCollection);

    TensorOffsetLayout targetOutLayout = valueTensor.createDefaultSubtensorLayout();
    targetOutLayout.shuffleDimensions(extractLayoutCollection);

    ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor(const_cast<Tensor&>(dataTensor), 0, 0, 0, targetInLayout.getDimensions()[0], targetInLayout);
    DAAL_CHECK_BLOCK_STATUS(dataSubtensor);
    const algorithmFPType *data = dataSubtensor.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> valueSubtensor(valueTensor, 0, 0, 0, targetOutLayout.getDimensions()[0], targetOutLayout);
    DAAL_CHECK_BLOCK_STATUS(valueSubtensor);
    algorithmFPType *value = valueSubtensor.get();

    WriteSubtensor<int, cpu, Tensor> selectedPosSubtensor;
    int *selectedPos = nullptr;
    size_t selectedPosSize = 0;
    bool trainingStage = !(parameter.predictionStage);
    if(trainingStage)
    {
        TensorOffsetLayout targetPosLayout = selectedPosTensor->createDefaultSubtensorLayout();
        targetPosLayout.shuffleDimensions(extractLayoutCollection);
        selectedPosSubtensor.set(*selectedPosTensor, 0, 0, 0, targetPosLayout.getDimensions()[0], targetPosLayout);
        DAAL_CHECK_BLOCK_STATUS(selectedPosSubtensor);
        selectedPos = selectedPosSubtensor.get();
        selectedPosSize = selectedPosTensor->getSize();
        invIntMaxVal = 1.0 / services::internal::MaxVal<int>::get();
    }

    const size_t nSlices = dataTensor.getSize() / (dims[par.firstIndex] * dims[par.secondIndex]);
    const size_t dataSliceSize = dims[par.firstIndex] * dims[par.secondIndex];
    const size_t valueSliceSize = valueDims[par.firstIndex] * valueDims[par.secondIndex];
    const size_t nFlatten = par.firstKernelSize * par.secondKernelSize;

    TArray<algorithmFPType, cpu> kernelWeightsPtr(nFlatten);
    algorithmFPType *kernelWeights = kernelWeightsPtr.get();
    DAAL_CHECK_MALLOC(kernelWeights);

    Status s;
    if(selectedPosSize != 0)
    {
        DAAL_CHECK_STATUS(s, getUniformRandFrom0to1(selectedPos, selectedPosSize, engine));
    }

    threader_for(nSlices, nSlices, [&](size_t i)
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
        for (DAAL_INT f = -par.firstPadding, fo = 0; fo < par.firstOutSize; f += par.firstStride, fo++)
        {
            /*
            * Loop by the second kernel dimension
            * s - index of the left upper corner of the kernel
            * so - index of the output value
            */
            for (DAAL_INT s = -par.secondPadding, so = 0; so < par.secondOutSize; s += par.secondStride, so++)
            {
                for(size_t l = 0; l < nFlatten; l++)
                {
                    kernelWeights[l] = (algorithmFPType)0.0;
                }

                algorithmFPType sum = (algorithmFPType)0.0;
                /*
                 * Loops over the kernel to get weights
                 */
                DAAL_INT fUpper = f + par.firstKernelSize;
                if (fUpper > par.firstSize + par.firstPadding)
                {
                    fUpper = par.firstSize + par.firstPadding;
                }
                for (DAAL_INT fi = f; fi < fUpper; fi++)
                {
                    DAAL_INT sUpper = s + par.secondKernelSize;
                    if (sUpper > par.secondSize + par.secondPadding)
                    {
                        sUpper = par.secondSize + par.secondPadding;
                    }
                    for (DAAL_INT si = s; si < sUpper; si++)
                    {
                        DAAL_INT dataIndex = si + par.secondSize * fi;
                        algorithmFPType dataValue = (par.getPaddingFlag(fi, si) ? 0.0 : dataSlice[dataIndex]);

                        sum += dataValue;
                        size_t l = (fi - f) * par.secondKernelSize + (si - s);
                        kernelWeights[l] = dataValue;
                    }
                }

                if (sum == 0.0)
                {
                    algorithmFPType invSum = 1.0 / par.firstKernelSize / par.secondKernelSize;
                    for(size_t l = 0; l < nFlatten; l++)
                    {
                        kernelWeights[l] = invSum;
                    }
                }
                else
                {
                    algorithmFPType invSum = 1.0 / sum;
                    for(size_t l = 0; l < nFlatten; l++)
                    {
                        kernelWeights[l] *= invSum;
                    }
                }

                DAAL_INT valueIndex = so + par.secondOutSize * fo;

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
    } );
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::computeWeightedAverage(
    const algorithmFPType *dataSlice,
    DAAL_INT f,
    DAAL_INT s,
    algorithmFPType *kernelWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value)
{
    value = 0;
    DAAL_INT fUpper = f + par.firstKernelSize;
    if (fUpper > par.firstSize + par.firstPadding)
    {
        fUpper = par.firstSize + par.firstPadding;
    }

    for (DAAL_INT fi = f; fi < fUpper; fi++)
    {
        DAAL_INT sUpper = s + par.secondKernelSize;
        if (sUpper > par.secondSize + par.secondPadding)
        {
            sUpper = par.secondSize + par.secondPadding;
        }

        for (DAAL_INT si = s; si < sUpper; si++)
        {
            DAAL_INT dataIndex = si + par.secondSize * fi;
            algorithmFPType dataValue = (par.getPaddingFlag(fi, si) ? 0.0 : dataSlice[dataIndex]);

            size_t l = (fi - f) * par.secondKernelSize + (si - s);
            value += kernelWeights[l] * dataValue;
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::getMultivariateRandomDataValue(
    const algorithmFPType *dataSlice,
    DAAL_INT f,
    DAAL_INT s,
    algorithmFPType *weights,
    size_t nWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value,
    int &selectedPos)
{
    selectedPos = getMultinomialRandomValue(weights, nWeights, selectedPos);
    size_t fi = f + (selectedPos / par.secondKernelSize);
    size_t si = s + (selectedPos - (fi - f) * par.secondKernelSize);

    DAAL_INT dataIndex = si + par.secondSize * fi;
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
Status PoolingKernel<algorithmFPType, method, cpu>::getUniformRandFrom0to1(int *uniformRand, const size_t nUniformRand, engines::BatchBase &engine)
{
    return distributions::uniform::internal::UniformKernelDefault<int, cpu>::compute(
        0, services::internal::MaxVal<int>::get(), engine, nUniformRand, uniformRand);
}

} // namespace internal
} // namespace forward
} // namespace stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
