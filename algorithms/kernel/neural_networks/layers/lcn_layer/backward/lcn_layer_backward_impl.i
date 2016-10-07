/* file: lcn_layer_backward_impl.i */
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

#include "convolution2d_layer_backward_kernel.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks::layers::convolution2d::backward::internal;

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
namespace backward
{
namespace internal
{
/*  step_1:   g_5   = inputGradient * auxInvMax;
    step_2:   sum_sumDimension( inputGradient * auxCenteredData );
    step_3:   g_13  = pow(auxInvMax, 2) * step_2;
    step_4:   g_10  = step_3 * q;
    step_5:   g_12  = g_13 * (1 - q) = g_13 - g_10 = step_3 - step_4;
    step_6:   g_8   = (g_10 + g_11) / auxSigma = (g_10 + 1/M * g_12) / auxSigma = (step_4 + 1/M * step_5) / auxSigma;
    step_7:   g_7   = dconv(g_8) = dconv(step_6);
    step_8:   g_4   = g_5 + g_6 = g_5 + g_7 * auxCenteredData = step_1 + step_7 * auxCenteredData;
    step_9:   g_3   = sum_sumDimension(g_4) = sum_sumDimension(step_8);
    step_10:  g_1   = dconv(g_3) = dconv(step_9);
    step_11:  gradient = g_2 - g_1 = g_4 - g_1 = step_8 - step_10.
*/
template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::compute(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter)
{
    /* Allocate arrays needed for computations */
    SmartPtr<cpu> gConvTempBlock( task.nSigmaElements * sizeof(algorithmFPType));
    algorithmFPType *gConvTempOfSigmaSizeArray = (algorithmFPType *)gConvTempBlock.get();
    if (!gConvTempOfSigmaSizeArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SmartPtr<cpu> gSqTempBlock( task.nSigmaElements * sizeof(algorithmFPType));
    algorithmFPType *gSqTempOfSigmaSizeArray = (algorithmFPType *)gSqTempBlock.get();
    if (!gSqTempOfSigmaSizeArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* steps 1-6 */
    prepareGDivGSqAndGConv(task, gConvTempOfSigmaSizeArray, gSqTempOfSigmaSizeArray);
    if (this->_errors->size() > 0) { return; }

    /* steps 7-11 */
    computeTwoConvAndFinalResult(task, parameter, gConvTempOfSigmaSizeArray, gSqTempOfSigmaSizeArray);
    if (this->_errors->size() > 0) { return; }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::prepareGDivGSqAndGConv(LCNTask<algorithmFPType, method, cpu> &task, algorithmFPType *gConvTempOfSigmaSizeArray,
                                                                 algorithmFPType *gSqTempOfSigmaSizeArray)
{
    const algorithmFPType one  = 1.0;
    const algorithmFPType zero = 0.0;

    SmartPtr<cpu> tempArrayOfCSizeBlock( task.nCElements * sizeof(algorithmFPType));
    algorithmFPType *tempArrayOfCSize = (algorithmFPType *)tempArrayOfCSizeBlock.get();
    if (!tempArrayOfCSize) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for(size_t i = 0; i < task.nSigmaElements; i++)
    {
        gSqTempOfSigmaSizeArray[i] = zero;
    }
    task.cBlock.release();

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
                /* step_1:   g_5 = inputGradient * auxInvMax */
                task.gradientArray[dataIndex] = task.inGradArray[dataIndex] * task.auxInvMaxArray[sigmaIndex];

                /* step_2:   sum_sumDimension( inputGradient * auxCenteredData ) */
                gSqTempOfSigmaSizeArray[sigmaIndex] += task.inGradArray[dataIndex] * task.auxCDArray[dataIndex];
            }
        }
    }
    task.inGradBlock.release();

    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < task.dataOffsetAfterDim; j++)
        {
            sigmaIndex = j + i * task.dataOffsetAfterDim;

            /* step_3:   g_13  = pow(auxInvMax, 2) * step_2 */
            gSqTempOfSigmaSizeArray[sigmaIndex] *= (- task.auxInvMaxArray[sigmaIndex] * task.auxInvMaxArray[sigmaIndex]);

            /* step_4:   g_10  = step_3 * q  */
            gConvTempOfSigmaSizeArray[sigmaIndex] = gSqTempOfSigmaSizeArray[sigmaIndex] * (task.auxSigmaArray[sigmaIndex] > task.auxCArray[i]);
        }
        tempArrayOfCSize[i] = zero;
    }

    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < task.dataOffsetAfterDim; j++)
        {
            sigmaIndex = j + i * task.dataOffsetAfterDim;
            /* step_5:   g_12  = g_13 * (1 - q) = g_13 - g_10 = step_3 - step_4 */
            tempArrayOfCSize[i] += gSqTempOfSigmaSizeArray[sigmaIndex] - gConvTempOfSigmaSizeArray[sigmaIndex];
        }
    }

    algorithmFPType divider = one / task.dataOffsetAfterDim;
    for(size_t i = 0; i < task.dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < task.dataOffsetAfterDim; j++)
        {
            sigmaIndex = j + i * task.dataOffsetAfterDim;
            /* step_6:  g_8  = (g_10 + g_11) / auxSigma = (g_10 + 1/M * g_12) / auxSigma = (step_4 + 1/M * step_5) / auxSigma */
            gSqTempOfSigmaSizeArray[sigmaIndex] = (gConvTempOfSigmaSizeArray[sigmaIndex] + divider * tempArrayOfCSize[i]) / (task.auxSigmaArray[sigmaIndex] + task.sigmaThreshold);
            gConvTempOfSigmaSizeArray[sigmaIndex] = zero;
        }
    }
    task.sigmaBlock.release();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::computeTwoConvAndFinalResult(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter,
                                                                              algorithmFPType *gConvTempOfSigmaSizeArray,
                                                                              algorithmFPType *gSqTempOfSigmaSizeArray)
{
    SmartPtr<cpu> weightsBlock( task.nWeightsElements * sizeof(algorithmFPType));
    algorithmFPType *weightsArray = (algorithmFPType *)weightsBlock.get();
    if (!weightsArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SmartPtr<cpu> wDerBlock( task.nWeightsElements * sizeof(algorithmFPType));
    algorithmFPType *wDerArray = (algorithmFPType *)wDerBlock.get();
    if (!wDerArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SmartPtr<cpu> convResultBlock( task.nDataElements * sizeof(algorithmFPType));
    algorithmFPType *convResultArray = (algorithmFPType *)convResultBlock.get();
    if (!convResultArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SmartPtr<cpu> bDerBlock( 1 * sizeof(algorithmFPType));
    algorithmFPType *bDerArray = (algorithmFPType *)bDerBlock.get();
    if (!bDerArray)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    services::Collection<size_t> bDims(1);
    bDims[0] = 1;

    /* Repack kernel to weightsArray needed for convolution computation */
    getConvolutionWeightsFromInputKernel(weightsArray, task, parameter);
    if (this->_errors->size() > 0) { return; }

    size_t nKernels = 1;
    size_t wDims[] = {nKernels, task.weightsSecondDim, task.kernelDims[0], task.kernelDims[1]};
    size_t convInpGradDims[] = {task.dataDims[0], 1, task.dataDims[task.firstDim], task.dataDims[task.secondDim]};

    SharedPtr<Tensor> weightsTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.nDims, wDims, weightsArray));
    if(weightsTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> convResultTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.dataDims, convResultArray)); /* is needed for weights and biases derivatives only, not used in lcn */
    if(convResultTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> inGradTensor  = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.nDims, convInpGradDims, gSqTempOfSigmaSizeArray));
    if(inGradTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> wDerTensor    = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.nDims, wDims, wDerArray)); /* only neede by conv compute, not lcn */
    if(wDerTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> bDerTensor    = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(bDims, bDerArray)); /* only neede by conv compute, not lcn */
    if(bDerTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Create backward convolution algorithm and set algorithm parameters */
    Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;

    /* step_7:  g_7  = dconv(g_8) = dconv(step_6) */
    /* convResultTensor first time is used here as auxData for wDer and bDer not needed calculation, second time as conv result, needed for lcn */
    dconvKernel.compute(inGradTensor.get(), convResultTensor.get(), weightsTensor.get(), &task.convParameter, wDerTensor.get(), bDerTensor.get(), convResultTensor.get());

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

                /* step_8:  g_4  = g_5 + g_6 = g_5 + g_7 * auxCenteredData = step_1 + step_7 * auxCenteredData */
                task.gradientArray[dataIndex] += convResultArray[dataIndex] * task.auxCDArray[dataIndex];

                /* step_9:  g_3  = sum_sumDimension(g_4) = sum_sumDimension(step_8) */
                gConvTempOfSigmaSizeArray[sigmaIndex] += task.gradientArray[dataIndex];
            }
        }
    }

    inGradTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(task.nDims, convInpGradDims, gConvTempOfSigmaSizeArray));
    if(inGradTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* step_10:  g_1  = dconv(g_3) = dconv(step_9) */
    dconvKernel.compute(inGradTensor.get(), convResultTensor.get(), weightsTensor.get(), &task.convParameter, wDerTensor.get(), bDerTensor.get(), convResultTensor.get());

    for(size_t i = 0; i < task.nDataElements; i++)
    {
        /* step_11:  gradient = g_2 - g_1 = g_4 - g_1 = step_8 - step_10 */
        task.gradientArray[i] -= convResultArray[i];
    }
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
    task.kernelBlock.release();
}

template<typename algorithmFPType, Method method, CpuType cpu>
LCNTask<algorithmFPType, method, cpu>::LCNTask(Tensor *auxCenteredDataTensor, Tensor *auxSigmaTensor,
                                               Tensor *auxCTensor, Tensor *auxInvMaxTensor, Tensor *kernelTensor, Tensor *inGradTensor,
                                               Tensor *gradientTensor, const lcn::Parameter *parameter) :
                                               inGradBlock(),
                                               cdBlock(),
                                               sigmaBlock(),
                                               cBlock(),
                                               gradientBlock(),
                                               invMaxBlock(),
                                               kernelBlock()
{
    services::Collection<size_t> initialDataDims = auxCenteredDataTensor->getDimensions();
    services::Collection<size_t> sigmaDims       = auxSigmaTensor->getDimensions();
    services::Collection<size_t> cDims           = auxCTensor->getDimensions();
    kernelDims = kernelTensor->getDimensions();

    size_t nDataRows   = initialDataDims[0];
    size_t nSigmaRows  = sigmaDims[0];
    size_t nCRows      = cDims[0];
    size_t nKernelRows = kernelDims[0];

    nSigmaElements  = auxSigmaTensor->getSize();
    nDataElements   = auxCenteredDataTensor->getSize();
    nKernelElements = kernelTensor->getSize();
    nCElements      = auxCTensor->getSize();
    nDims = initialDataDims.size();

    size_t initialFirstDim  = parameter->indices.dims[0];
    size_t initialSecondDim = parameter->indices.dims[1];

    sigmaThreshold = parameter->sigmaDegenerateCasesThreshold;

    size_t initialSumDimension = 1;
    if(parameter->sumDimension)
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

    const size_t dimsArray[4]  = {batchDimension, initialSumDimension, initialFirstDim, initialSecondDim };

    TensorOffsetLayout cdLayout = auxCenteredDataTensor->createDefaultSubtensorLayout();
    cdLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    TensorOffsetLayout inGradLayout = inGradTensor->createDefaultSubtensorLayout();
    inGradLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    TensorOffsetLayout gradientLayout = gradientTensor->createDefaultSubtensorLayout();
    gradientLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    inGradBlock.set(*inGradTensor, 0, 0, 0, nSigmaRows, inGradLayout);
    inGradArray = inGradBlock.get();

    cdBlock.set(*auxCenteredDataTensor, 0, 0, 0, nDataRows, cdLayout);
    auxCDArray = cdBlock.get();

    sigmaBlock.set(*auxSigmaTensor, 0, 0, 0, nSigmaRows);
    auxSigmaArray = sigmaBlock.get();

    invMaxBlock.set(*auxInvMaxTensor, 0, 0, 0, nSigmaRows);
    auxInvMaxArray = invMaxBlock.get();

    cBlock.set(*auxCTensor, 0, 0, 0, nCRows);
    auxCArray = cBlock.get();

    gradientBlock.set(*gradientTensor, 0, 0, 0, nDataRows, gradientLayout);
    gradientArray = gradientBlock.get();

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
} // backward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
