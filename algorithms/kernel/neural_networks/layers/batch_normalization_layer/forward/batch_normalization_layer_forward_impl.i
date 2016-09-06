/* file: batch_normalization_layer_forward_impl.i */
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
//  Implementation of forward batch normalization layer
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__

#include "service_math.h"

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::compute(
            const batch_normalization::forward::Input *input,
            const batch_normalization::Parameter *parameter,
            batch_normalization::forward::Result *result)
{
    size_t dimension = parameter->dimension;

    BatchNormalizationTask<algorithmFPType, method, cpu> task(input, result, dimension);

    /* Compute mini-batch mean and variance */
    computeMeanAndVariance(task.data, task.offsetBefore, task.dimensionSize, task.offsetAfter,
        task.mean, task.stDev);

    if(parameter->predictionStage == false)
    {
        /* Update population mean and variance */
        algorithmFPType alpha = (algorithmFPType)(parameter->alpha);
        updatePopulationMeanAndVariance(input, task.dimensionSize, alpha, task.mean, task.stDev, result);
    }

    /* Compute mini-batch standard deviation */
    algorithmFPType epsilon = (algorithmFPType)(parameter->epsilon);
    algorithmFPType *stDev = task.stDev;
    for (size_t k = 0; k < task.dimensionSize; k++)
    {
        stDev[k] += epsilon;
    }
    daal::internal::Math<algorithmFPType,cpu>::vSqrt(task.dimensionSize, stDev, stDev);

    /* Compute resulting value */
    computeResult(input, task.data, task.offsetBefore, task.dimensionSize, task.offsetAfter,
        task.mean, task.stDev, result);
}

/**
 * Computes mini-batch mean and variance of the input data over the specified dimension k
 *
 * Input data tensor is viewed by this method as a 3-dimensional tensor of size offsetBefore * dimensionSize * offsetAfter
 *
 * \param[in] data          Buffer that contains a block of values from input p-dimensional data tensor
 *                          of size n_1 * ... * n_p
 * \param[in] offsetBefore  n_1 * ... * n_(k-1)
 * \param[in] dimensionSize n_k - size of the dimension over which the mini-batch mean and variance are computed
 * \param[in] offsetAfter   n_(k+1) * ... * n_p
 * \param[out] mean         Resulting mean vector computed over the dimension k
 * \param[out] variance     Resulting variance vector computed over the dimension k
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::computeMeanAndVariance(
            const algorithmFPType *data, size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
            algorithmFPType *mean, algorithmFPType *variance)
{
    size_t m = offsetBefore * offsetAfter;
    algorithmFPType invM  = 1.0 / (algorithmFPType)m;
    algorithmFPType invM1 = 1.0 / (algorithmFPType)(m - 1);

    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            algorithmFPType sum = 0.0;
            algorithmFPType sumSq = 0.0;
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                algorithmFPType value = data[index];
                sum += value;
                sumSq += value * value;
            }
            mean[k]  += sum;
            variance[k] += sumSq;
        }
    }

    for (size_t k = 0; k < dimensionSize; k++)
    {
        variance[k] = invM1 * (variance[k] - mean[k] * mean[k] * invM);
        mean[k] *= invM;
    }
}

/**
 * Updates population mean and variance with the mini-batch mean and variance
 *
 * \param[in] input         Input object for the forward batch normalization layer containing
 *                          the values of population mean and variance from the previous iteration
 * \param[in] dimensionSize n_k - size of the dimension over which the mini-batch mean and variance are computed
 * \param[in] alpha         Smoothing factor that is used in population mean and population variance computations
 * \param[in] mean          Mini-batch mean computed over the dimension k
 * \param[in] variance      Mini-batch variance vector computed over the dimension k
 * \param[out] result       Result of the forward batch normalization layer containing
 *                          the updated values of population mean and variance
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::updatePopulationMeanAndVariance(
            const batch_normalization::forward::Input *input, size_t dimensionSize, algorithmFPType alpha,
            algorithmFPType *mean, algorithmFPType *variance, batch_normalization::forward::Result *result)
{
    SharedPtr<Tensor> inputPopulationMeanTensor     = input->get(populationMean);
    SharedPtr<Tensor> inputPopulationVarianceTensor = input->get(populationVariance);

    SharedPtr<Tensor> populationMeanTensor     = result->get(auxPopulationMean);
    SharedPtr<Tensor> populationVarianceTensor = result->get(auxPopulationVariance);

    SubtensorDescriptor<algorithmFPType> inputPopulationMeanBlock, inputPopulationVarianceBlock;
    SubtensorDescriptor<algorithmFPType> populationMeanBlock, populationVarianceBlock;

    inputPopulationMeanTensor    ->getSubtensor(0, 0, 0, dimensionSize, readOnly, inputPopulationMeanBlock);
    inputPopulationVarianceTensor->getSubtensor(0, 0, 0, dimensionSize, readOnly, inputPopulationVarianceBlock);

    populationMeanTensor    ->getSubtensor(0, 0, 0, dimensionSize, writeOnly, populationMeanBlock);
    populationVarianceTensor->getSubtensor(0, 0, 0, dimensionSize, writeOnly, populationVarianceBlock);

    algorithmFPType *inputPopulationMeanArray     = inputPopulationMeanBlock.getPtr();
    algorithmFPType *inputPopulationVarianceArray = inputPopulationVarianceBlock.getPtr();
    algorithmFPType *populationMeanArray     = populationMeanBlock.getPtr();
    algorithmFPType *populationVarianceArray = populationVarianceBlock.getPtr();

    for (size_t i = 0; i < dimensionSize; i++)
    {
        populationMeanArray[i]     = inputPopulationMeanArray[i];
        populationVarianceArray[i] = inputPopulationVarianceArray[i];
    }

    inputPopulationMeanTensor    ->releaseSubtensor(inputPopulationMeanBlock);
    inputPopulationVarianceTensor->releaseSubtensor(inputPopulationVarianceBlock);

    for (size_t i = 0; i < dimensionSize; i++)
    {
        populationMeanArray[i]     += alpha * mean[i];
        populationVarianceArray[i] += alpha * variance[i];
    }

    populationMeanTensor    ->releaseSubtensor(populationMeanBlock);
    populationVarianceTensor->releaseSubtensor(populationVarianceBlock);
}

/**
 * Computes batch normalization results
 *
 * Input data tensor is viewed by this method as a 3-dimensional tensor of size offsetBefore * dimensionSize * offsetAfter
 *
 * \param[in] input         Input object for the forward batch normalization layer containing
 *                          weights and biases for the batch normalization
 * \param[in] data          Buffer that contains a block of values from input p-dimensional data tensor
 *                          of size n_1 * ... * n_p
 * \param[in] offsetBefore  n_1 * ... * n_(k-1)
 * \param[in] dimensionSize n_k - size of the dimension over which the mini-batch mean and variance are computed
 * \param[in] offsetAfter   n_(k+1) * ... * n_p
 * \param[in] mean          Mini-batch mean vector computed over the dimension k
 * \param[in] stDev         Mini-batch standard deviation vector computed over the dimension k
 * \param[out] result       Result of the forward batch normalization layer
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void BatchNormalizationKernel<algorithmFPType, method, cpu>::computeResult(
            const batch_normalization::forward::Input *input,
            const algorithmFPType *data, size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
            const algorithmFPType *mean, const algorithmFPType *stDev,
            batch_normalization::forward::Result *result)
{
    SharedPtr<Tensor> weightsTensor = input->get(layers::forward::weights);
    SharedPtr<Tensor> biasesTensor  = input->get(layers::forward::biases);
    SharedPtr<Tensor> valueTensor = result->get(layers::forward::value);

    SubtensorDescriptor<algorithmFPType> weightsBlock, biasesBlock, valueBlock;
    weightsTensor->getSubtensor(0, 0, 0, dimensionSize, readOnly, weightsBlock);
    biasesTensor ->getSubtensor(0, 0, 0, dimensionSize, readOnly, biasesBlock);
    valueTensor  ->getSubtensor(0, 0, 0, (valueTensor->getDimensions())[0], writeOnly, valueBlock);

    algorithmFPType *weightsArray = weightsBlock.getPtr();
    algorithmFPType *biasesArray  = biasesBlock.getPtr();
    algorithmFPType *valueArray   = valueBlock.getPtr();

    algorithmFPType *invStDevByWeights = (algorithmFPType *)daal_malloc(dimensionSize * sizeof(algorithmFPType));
    if (!invStDevByWeights) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for (size_t k = 0; k < dimensionSize; k++)
    {
        invStDevByWeights[k] = weightsArray[k] / stDev[k];
    }

    for (size_t i = 0; i < offsetBefore; i++)
    {
        for (size_t k = 0; k < dimensionSize; k++)
        {
            for (size_t j = 0; j < offsetAfter; j++)
            {
                size_t index = (i * dimensionSize + k) * offsetAfter + j;
                valueArray[index] = invStDevByWeights[k] * (data[index] - mean[k]) + biasesArray[k];
            }
        }
    }

    weightsTensor->releaseSubtensor(weightsBlock);
    biasesTensor ->releaseSubtensor(biasesBlock);
    valueTensor  ->releaseSubtensor(valueBlock);
    daal_free(invStDevByWeights);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchNormalizationTask<algorithmFPType, method, cpu>::BatchNormalizationTask(
            const batch_normalization::forward::Input *input,
            batch_normalization::forward::Result *result,
            size_t dimension):
     inputTensor(input->get(layers::forward::data)),
     meanTensor (result->get(auxMean)),
     stDevTensor(result->get(auxStandardDeviation))
{
    const services::Collection<size_t>& dims = inputTensor->getDimensions();
    dimensionSize = dims[dimension];

    inputTensor->getSubtensor(0, 0, 0, dims[0], readOnly, inputBlock);
    meanTensor ->getSubtensor(0, 0, 0, dimensionSize, writeOnly, meanBlock);
    stDevTensor->getSubtensor(0, 0, 0, dimensionSize, writeOnly, stDevBlock);

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

    data = inputBlock.getPtr();
    mean  = meanBlock.getPtr();
    stDev = stDevBlock.getPtr();

    /* Initialize mean and stDev arrays */
    const algorithmFPType zero = (algorithmFPType)0.0;
    for (size_t i = 0; i < dimensionSize; i++)
    {
        mean[i] = zero;
        stDev[i] = zero;
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchNormalizationTask<algorithmFPType, method, cpu>::~BatchNormalizationTask()
{
    meanTensor->releaseSubtensor(meanBlock);
    stDevTensor->releaseSubtensor(stDevBlock);
    inputTensor->releaseSubtensor(inputBlock);
}

} // namespace internal
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
