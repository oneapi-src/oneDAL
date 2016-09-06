/* file: lcn_layer_forward_impl.i */
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
//  Implementation of local contrast normalization algorithm
//--
*/

#include "convolution2d_layer_forward_kernel.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::layers::convolution2d::forward::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{
namespace forward
{
namespace internal
{
/*  step_1:  x_3  = conv(data);
    step_2:  x_4  = centeredData = data - step_1;
    step_3:  x_7  = pow(step_2, 2);
    step_4:  x_8  = conv(step_3);
    step_5:  x_9  = sigma = sqrt(step_4);
    step_6:  x_12 = c = mean(step_5);
    step_7:  1/x_13 = 1 / max(step_5, step_6);
    step_8: result  = step_2 * step_7.
*/
template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::compute(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter)
{
    /* steps 1-5 */
    calculateCenteredDataAndSigma(task, parameter);
    /* steps 6-7 */
    calculateCTensorAndGetMaxArray(task);
    /* step 8 */
    calculateResult(task);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::getConvolutionWeightsFromInputKernel(algorithmFPType *weightsArray, LCNTask<algorithmFPType, method, cpu> &task,
        const lcn::Parameter *parameter)
{
    if(parameter->sumDimension)
    {
        /* copy input kernel dataDims[sumDimension] times to get weights for convolution and normalize through sumDimension */
        algorithmFPType divider = (algorithmFPType)1.0 / task.dataDims[task.sumDimension];

        for(size_t i = 0; i < task.dataDims[task.sumDimension]; i++)
        {
            for(size_t j = 0; j < task.nKernelElements; j++)
            {
                weightsArray[j + i * task.nKernelElements] = task.kernelArray[j] * divider;
            }
        }
    }
    else
    {
        /* Copy input kernel to convolution weights */
        for(size_t j = 0; j < task.nKernelElements; j++)
        {
            weightsArray[j] = task.kernelArray[j];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateCenteredDataAndSigma(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter)
{
    Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> convKernel;

    /* Create forward convolution algorithm and set algorithm parameters */
    if (this->_errors->size() > 0) { return; }

    SmartPtr<cpu> weightsBlock( task.nWeightsElements * sizeof(algorithmFPType));
    algorithmFPType *weightsArray = (algorithmFPType *)weightsBlock.get();
    if (!weightsArray)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    getConvolutionWeightsFromInputKernel(weightsArray, task, parameter);
    if (this->_errors->size() > 0) { return; }

    SmartPtr<cpu> biasesBlock( 1 * sizeof(algorithmFPType));
    algorithmFPType *biasesArray = (algorithmFPType *)biasesBlock.get();
    if (!biasesArray)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    size_t nKernels = 1;
    size_t wDims[] = {nKernels, task.weightsSecondDim, task.kernelDims[0], task.kernelDims[1]};
    size_t sDims[] = {task.dataDims[0], 1, task.dataDims[2], task.dataDims[3]};
    SharedPtr<Tensor> weightsTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.dataDims.size(), wDims, weightsArray));
    if(weightsTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Fill null convolution biases */
    services::Collection<size_t> bDims(1);
    bDims[0] = nKernels;
    biasesArray[0] = (algorithmFPType)0;

    SharedPtr<Tensor> inputTensor  = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.dataDims, const_cast<algorithmFPType *>(task.inputArray)));
    if(inputTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> biasesTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(bDims, biasesArray));
    if(biasesTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> valueTensor  = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(4, sDims, const_cast<algorithmFPType *>(task.sigmaArray)));
    if(valueTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* step_1:  x_3 = conv(data) */
    convKernel.compute(inputTensor.get(), weightsTensor.get(), biasesTensor.get(), &task.convParameter, valueTensor.get());

    size_t dataIndex = 0;
    size_t sigmaIndex = 0;
    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < task.dataDims[task.sumDimension]; j++)
        {
            for(size_t k = 0; k < task.dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j * task.dataOffsetAfterDim + i * task.dataDims[task.sumDimension] * task.dataOffsetAfterDim;
                sigmaIndex = k + i * task.dataOffsetAfterDim;
                /* step_2:  x_4 = centeredData = data - step_1 */
                task.centeredDataArray[dataIndex] = task.inputArray[dataIndex] - task.sigmaArray[sigmaIndex];
            }
        }
    }
    task.inputBlock.release();

    /* step_3:  x_7 = pow(step_2, 2) */
    daal::internal::Math<algorithmFPType, cpu>::vPowx(task.nDataElements, task.centeredDataArray, (algorithmFPType)2, task.resultArray);

    inputTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.dataDims, const_cast<algorithmFPType *>(task.resultArray)));
    if(inputTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* step_4:  x_8 = conv(step_3) */
    convKernel.compute(inputTensor.get(), weightsTensor.get(), biasesTensor.get(), &task.convParameter, valueTensor.get());

    /* step_5:  x_9 = sigma = sqrt(step_4) */
    daal::internal::Math<algorithmFPType, cpu>::vSqrt(task.nSigmaElements, task.sigmaArray, task.sigmaArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateCTensorAndGetMaxArray(LCNTask<algorithmFPType, method, cpu> &task)
{
    const algorithmFPType one = 1.0;
    const algorithmFPType zero = 0;

    algorithmFPType divider = one / task.dataOffsetAfterDim;

    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        task.cArray[i] = zero;
        for(size_t j = 0; j < task.dataOffsetAfterDim; j++)
        {
            /* step_6:  x_12 = c = mean(step_5) */
            task.cArray[i] += task.sigmaArray[j + i * task.dataOffsetAfterDim];
        }

        task.cArray[i] *= divider;

        for(size_t j = 0; j < task.dataOffsetAfterDim; j++)
        {
            /* step_7:  1/x_13 = 1 / max(step_5, step_6) */
            task.invMaxArray[j + i * task.dataOffsetAfterDim] = one / daal::internal::Math<algorithmFPType, cpu>::sMax(task.sigmaArray[j + i * task.dataOffsetAfterDim], task.cArray[i]);
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateResult(LCNTask<algorithmFPType, method, cpu> &task)
{
    size_t dataIndex = 0;
    size_t sigmaIndex = 0;
    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < task.dataDims[task.sumDimension]; j++)
        {
            for(size_t k = 0; k < task.dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j * task.dataOffsetAfterDim + i * task.dataDims[task.sumDimension] * task.dataOffsetAfterDim;
                sigmaIndex = k + i * task.dataOffsetAfterDim;
                /* step_8: result = step_2 * step_7  */
                task.resultArray[dataIndex] = task.centeredDataArray[dataIndex] * task.invMaxArray[sigmaIndex];
            }
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
LCNTask<algorithmFPType, method, cpu>::LCNTask(Tensor *inputTensor, Tensor *resultTensor, Tensor *centeredDataTensor,
        Tensor *sigmaTensor, Tensor *cTensor, Tensor *invMaxTensor, const lcn::Parameter *parameter,
        Tensor *kernelTensor) :
    inputBlock(),
    resultBlock(),
    cdBlock(),
    cBlock(),
    invMaxBlock(),
    kernelBlock()
{
    services::Collection<size_t> initialDataDims = inputTensor->getDimensions();
    services::Collection<size_t> sigmaDims       = invMaxTensor->getDimensions();
    services::Collection<size_t> cDims           = cTensor->getDimensions();
    kernelDims = kernelTensor->getDimensions();

    size_t nInputRows  = initialDataDims[0];
    size_t nSigmaRows  = sigmaDims[0];
    size_t nCRows      = cDims[0];
    size_t nKernelRows = kernelDims[0];

    nSigmaElements  = invMaxTensor->getSize();
    nDataElements   = inputTensor->getSize();
    nKernelElements = kernelTensor->getSize();

    size_t initialFirstDim  = parameter->indices.dims[0];
    size_t initialSecondDim = parameter->indices.dims[1];

    size_t initialSumDimension = 1;
    if(parameter->sumDimension) /* initialDimension is needed here only for getting correct inputArray */
    {
        ReadRows<algorithmFPType, cpu, NumericTable> dimBlock(*parameter->sumDimension, 0, 1);
        const algorithmFPType *dimArray = dimBlock.get();
        initialSumDimension = (size_t)dimArray[0];
    }

    /* Get dims collection of repacked data tensor */
    size_t batchDimension = 6 - initialSumDimension - initialFirstDim - initialSecondDim; /* Calculate 4th dimension index. 6 here is a sum of all indexes: 0 + 1 + 2 + 3 */
    dataDims.push_back(initialDataDims[batchDimension]);
    dataDims.push_back(initialDataDims[initialSumDimension]);
    dataDims.push_back(initialDataDims[initialFirstDim]);
    dataDims.push_back(initialDataDims[initialSecondDim]);

    const size_t dimsArray[4] = {batchDimension, initialSumDimension, initialFirstDim, initialSecondDim};

    TensorOffsetLayout inputLayout = inputTensor->createDefaultSubtensorLayout();
    inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    TensorOffsetLayout cdLayout = centeredDataTensor->createDefaultSubtensorLayout();
    cdLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    TensorOffsetLayout resultLayout = resultTensor->createDefaultSubtensorLayout();
    resultLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    inputBlock.set(*inputTensor, 0, 0, 0, nInputRows, inputLayout);
    inputArray = inputBlock.get(); /* already repacked array that has 1st dim as sumDimension, 2nd as firstDim and 3rd as secondDim */

    resultBlock.set(*resultTensor, 0, 0, 0, nInputRows, resultLayout);
    resultArray = resultBlock.get();

    cdBlock.set(*centeredDataTensor, 0, 0, 0, nInputRows, cdLayout);
    centeredDataArray = cdBlock.get();

    cBlock.set(*cTensor, 0, 0, 0, nCRows);
    cArray = cBlock.get();

    invMaxBlock.set(*invMaxTensor, 0, 0, 0, nSigmaRows);
    invMaxArray = invMaxBlock.get();

    if(parameter->predictionStage == false)
    {
        sigmaBlock.set(*sigmaTensor, 0, 0, 0, nSigmaRows);
        sigmaArray = sigmaBlock.get();
    }
    else
    {
        sigmaArray = invMaxArray; // when we on prediction stage, we need not compute invMaxArray, and we don't use then simultaneously, so they refer on the same memory
    }

    kernelBlock.set(*kernelTensor, 0, 0, 0, nKernelRows);
    kernelArray = kernelBlock.get();

    sumDimension = (size_t)1;
    firstDim     = (size_t)2;
    secondDim    = (size_t)3;

    if(parameter->sumDimension)
    {
        weightsSecondDim = dataDims[sumDimension];
    }
    else
    {
        weightsSecondDim = 1;
        dataDims[0] *= dataDims[sumDimension];
        dataDims[sumDimension] = 1;
    }

    dataOffsetBeforeDim   = dataDims[0];
    dataOffsetAfterDim    = dataDims[firstDim] * dataDims[secondDim];
    nWeightsElements      = dataDims[sumDimension] * nKernelElements;

    /* Set convolution algorithm parameters */
    convParameter.indices.dims[0] = firstDim;
    convParameter.indices.dims[1] = secondDim;
    convParameter.nGroups = 1;
    convParameter.strides.size[0] = 1;
    convParameter.strides.size[1] = 1;
    convParameter.groupDimension = 1;
    convParameter.nKernels = 1;
    convParameter.kernelSizes.size[0] = kernelDims[0];
    convParameter.kernelSizes.size[1] = kernelDims[1];
    convParameter.paddings.size[0] = kernelDims[0] / 2;
    convParameter.paddings.size[1] = kernelDims[1] / 2;
}

} // internal
} // forward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
