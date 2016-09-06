/* file: batch_normalization_layer_backward_impl.i */
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
//  Implementation of backward batch normalization layer
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_IMPL_I__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::compute(
            const batch_normalization::backward::Input *input, const batch_normalization::Parameter *parameter,
            batch_normalization::backward::Result *result)
{
    size_t dimension = parameter->dimension;

    BatchNormalizationTask<algorithmFPType, method, cpu> task(input, result, dimension, this->_errors);

    computeWeightsAndBiasesDerivatives(task.inputGradient, task.data,
        task.offsetBefore, task.dimensionSize, task.offsetAfter, task.mean, task.invStDev,
        task.weightsDer, task.biasesDer);

    computeGradient(input, task.inputGradient, task.data,
        task.offsetBefore, task.dimensionSize, task.offsetAfter, task.mean, task.invStDev,
        task.weightsDer, task.biasesDer, result);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::computeWeightsAndBiasesDerivatives(
            const algorithmFPType *inputGradient, const algorithmFPType *data,
            size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
            const algorithmFPType *mean, const algorithmFPType *invStDev,
            algorithmFPType *weightsDer, algorithmFPType *biasesDer)
{
    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            algorithmFPType weightsDerSum = 0.0;
            algorithmFPType biasesDerSum = 0.0;
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                biasesDerSum += inputGradient[index];
                weightsDerSum += inputGradient[index] * (data[index] - mean[k]) * invStDev[k];
            }
            biasesDer[k] += biasesDerSum;
            weightsDer[k] += weightsDerSum;
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::computeGradient(
            const batch_normalization::backward::Input *input,
            const algorithmFPType *inputGradient, const algorithmFPType *data,
            size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
            const algorithmFPType *mean, const algorithmFPType *invStDev,
            const algorithmFPType *weightsDer, const algorithmFPType *biasesDer,
            batch_normalization::backward::Result *result)
{
    size_t m = offsetBefore * offsetAfter;
    algorithmFPType invM  = 1.0 / (algorithmFPType)m;
    algorithmFPType invM1 = 1.0 / (algorithmFPType)(m - 1);

    SharedPtr<Tensor> gradientTensor = result->get(layers::backward::gradient);
    SubtensorDescriptor<algorithmFPType> gradientBlock;
    algorithmFPType *gradient;

    SharedPtr<Tensor> weightsTensor = input->get(auxWeights);
    SubtensorDescriptor<algorithmFPType> weightsBlock;
    algorithmFPType *weights;

    const services::Collection<size_t>& dims = gradientTensor->getDimensions();
    gradientTensor->getSubtensor(0, 0, 0, dims[0], writeOnly, gradientBlock);
    weightsTensor->getSubtensor(0, 0, 0, dimensionSize, readOnly, weightsBlock);
    gradient = gradientBlock.getPtr();
    weights = weightsBlock.getPtr();

    algorithmFPType *invStDevByWeights    = (algorithmFPType *)daal_malloc(dimensionSize * sizeof(algorithmFPType));
    algorithmFPType *biasesDerMultiplier  = (algorithmFPType *)daal_malloc(dimensionSize * sizeof(algorithmFPType));
    algorithmFPType *weightsDerMultiplier = (algorithmFPType *)daal_malloc(dimensionSize * sizeof(algorithmFPType));
    if (!invStDevByWeights || !biasesDerMultiplier || !weightsDerMultiplier)
    { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for (size_t k = 0; k < dimensionSize; k++)
    {
        invStDevByWeights[k] = weights[k] * invStDev[k];
        biasesDerMultiplier[k] = invM * biasesDer[k];
        weightsDerMultiplier[k] = invM1 * invStDev[k] * weightsDer[k];
    }

    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                gradient[index] = invStDevByWeights[k] *
                    (inputGradient[index] - biasesDerMultiplier[k] - weightsDerMultiplier[k] * (data[index] - mean[k]));
            }
        }
    }

    gradientTensor->releaseSubtensor(gradientBlock);
    weightsTensor->releaseSubtensor(weightsBlock);
    daal_free(invStDevByWeights);
    daal_free(biasesDerMultiplier);
    daal_free(weightsDerMultiplier);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchNormalizationTask<algorithmFPType, method, cpu>::BatchNormalizationTask(
            const batch_normalization::backward::Input *input,
            batch_normalization::backward::Result *result,
            size_t dimension, services::SharedPtr<services::KernelErrorCollection> _errors):
     inputGradientTensor(input->get(layers::backward::inputGradient)),
     dataTensor (input->get(auxData)),
     meanTensor (input->get(auxMean)),
     weightsDerTensor(result->get(layers::backward::weightDerivatives)),
     biasesDerTensor(result->get(layers::backward::biasDerivatives))
{
    SharedPtr<Tensor> stDevTensor = input->get(auxStandardDeviation);
    SubtensorDescriptor<algorithmFPType> stDevBlock;
    algorithmFPType *stDev;

    const services::Collection<size_t>& dims = inputGradientTensor->getDimensions();
    dimensionSize = dims[dimension];

    inputGradientTensor->getSubtensor(0, 0, 0, dims[0], readOnly, inputGradientBlock);
    dataTensor->getSubtensor(0, 0, 0, dims[0], readOnly, dataBlock);
    meanTensor->getSubtensor(0, 0, 0, dimensionSize, readOnly, meanBlock);
    stDevTensor->getSubtensor(0, 0, 0, dimensionSize, readOnly, stDevBlock);
    weightsDerTensor->getSubtensor(0, 0, 0, dimensionSize, writeOnly, weightsDerBlock);
    biasesDerTensor->getSubtensor(0, 0, 0, dimensionSize, writeOnly, biasesDerBlock);

    offsetBefore = 1;
    offsetAfter = 1;
    for (size_t i = 0; i < dimension; i++)
    {
        offsetBefore *= dims[i];
    }
    for (size_t i = dimension + 1; i < dims.size(); i++)
    {
        offsetAfter *= dims[i];
    }

    inputGradient = inputGradientBlock.getPtr();
    data  = dataBlock.getPtr();
    mean  = meanBlock.getPtr();
    stDev = stDevBlock.getPtr();
    weightsDer = weightsDerBlock.getPtr();
    biasesDer = biasesDerBlock.getPtr();

    invStDev = (algorithmFPType *)daal_malloc(dimensionSize * sizeof(algorithmFPType));
    if (!invStDev) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Initialize inverse standard deviation */
    const algorithmFPType one = (algorithmFPType)1.0;
    for (size_t i = 0; i < dimensionSize; i++)
    {
        invStDev[i] = one / stDev[i];
    }

    /* Initialize weights and biases derivatives */
    const algorithmFPType zero = (algorithmFPType)0.0;
    for (size_t i = 0; i < dimensionSize; i++)
    {
        weightsDer[i] = zero;
        biasesDer[i] = zero;
    }
    stDevTensor->releaseSubtensor(stDevBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchNormalizationTask<algorithmFPType, method, cpu>::~BatchNormalizationTask()
{
    inputGradientTensor->releaseSubtensor(inputGradientBlock);
    dataTensor->releaseSubtensor(dataBlock);
    meanTensor->releaseSubtensor(meanBlock);
    weightsDerTensor->releaseSubtensor(weightsDerBlock);
    biasesDerTensor->releaseSubtensor(biasesDerBlock);
    daal_free(invStDev);
}

} // namespace internal
} // namespace backward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
