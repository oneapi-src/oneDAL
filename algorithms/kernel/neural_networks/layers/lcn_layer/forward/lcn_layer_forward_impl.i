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
void LCNKernel<algorithmFPType, method, cpu>::initialize(Tensor *inputTensor, Tensor *cTensor, Tensor *invMaxTensor,
                                                         const lcn::Parameter *parameter, Tensor *kernelTensor)
{
    services::Collection<size_t> initialDataDims = inputTensor->getDimensions();
    services::Collection<size_t> sigmaDims       = invMaxTensor->getDimensions();
    services::Collection<size_t> cDims           = cTensor->getDimensions();
    kernelDims = kernelTensor->getDimensions();

    nInputRows  = initialDataDims[0];
    nSigmaRows  = sigmaDims[0];
    nCRows      = cDims[0];
    nKernelRows = kernelDims[0];

    nSigmaElements  = invMaxTensor->getSize();
    nDataElements   = inputTensor->getSize();
    nKernelElements = kernelTensor->getSize();

    initialFirstDim  = parameter->indices.dims[0];
    initialSecondDim = parameter->indices.dims[1];

    initialSumDimension = 1;
    if(parameter->sumDimension) /* initialDimension is needed here only for getting correct inputArray */
    {
        ReadRows<algorithmFPType, cpu, NumericTable> dimBlock(*parameter->sumDimension, 0, 1);
        const algorithmFPType *dimArray = dimBlock.get();
        initialSumDimension = (size_t)dimArray[0];
    }

    /* Get dims collection of repacked data tensor */
    batchDimension = 6 - initialSumDimension - initialFirstDim - initialSecondDim; /* Calculate 4th dimension index. 6 here is a sum of all indexes: 0 + 1 + 2 + 3 */
    dataDims << initialDataDims[batchDimension] << initialDataDims[initialSumDimension] << initialDataDims[initialFirstDim] << initialDataDims[initialSecondDim];

    dimsArray[0] = {batchDimension};
    dimsArray[1] = {initialSumDimension};
    dimsArray[2] = {initialFirstDim};
    dimsArray[3] = {initialSecondDim};

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

    nKernels = 1;

    weightsDims << nKernels << weightsSecondDim << kernelDims[0] << kernelDims[1];
    sDims << dataDims[0] << 1 << dataDims[2] << dataDims[3];

    /* Set convolution algorithm parameters */
    convParameter.indices.dims[0] = firstDim;
    convParameter.indices.dims[1] = secondDim;
    convParameter.nGroups = 1;
    convParameter.strides.size[0] = 1;
    convParameter.strides.size[1] = 1;
    convParameter.groupDimension = 1;
    convParameter.nKernels = nKernels;
    convParameter.kernelSizes.size[0] = kernelDims[0];
    convParameter.kernelSizes.size[1] = kernelDims[1];
    convParameter.paddings.size[0] = kernelDims[0] / 2;
    convParameter.paddings.size[1] = kernelDims[1] / 2;

    convKernel.initialize(dataDims, weightsDims, &convParameter, sDims);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor,  Tensor *sigmaTensor, Tensor *cTensor, Tensor *resultTensor, Tensor *centeredDataTensor,
                                                      Tensor *invMaxTensor, const lcn::Parameter *parameter, Tensor *kernelTensor)
{
    TensorOffsetLayout cdLayout = centeredDataTensor->createDefaultSubtensorLayout();
    cdLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    TensorOffsetLayout resultLayout = resultTensor->createDefaultSubtensorLayout();
    resultLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, 0, 0, 0, nInputRows, resultLayout);
    resultArray = resultBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> cdBlock(*centeredDataTensor, 0, 0, 0, nInputRows, cdLayout);
    centeredDataArray = cdBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> invMaxBlock(*invMaxTensor, 0, 0, 0, nSigmaRows);
    invMaxArray = invMaxBlock.get();

    if(parameter->predictionStage == false)
    {
        WriteSubtensor<algorithmFPType, cpu, Tensor> sigmaBlock(*sigmaTensor, 0, 0, 0, nSigmaRows);
        sigmaArray = sigmaBlock.get();
    }
    else
    {
        sigmaArray = invMaxArray; /* when we on prediction stage, we need not compute invMaxArray, and we don't use then simultaneously, so they refer on the same memory */
    }

    ReadSubtensor<algorithmFPType, cpu, Tensor> kernelBlock(*kernelTensor, 0, 0, 0, nKernelRows);
    kernelArray = kernelBlock.get();

    /* steps 1-5 */
    calculateCenteredDataAndSigma(parameter, inputTensor);
    /* steps 6-7 */
    calculateCTensorAndGetMaxArray(cTensor);
    /* step 8 */
    calculateResult();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::getConvolutionWeightsFromInputKernel(algorithmFPType *weightsArray,
        const lcn::Parameter *parameter)
{
    if(parameter->sumDimension)
    {
        /* copy input kernel dataDims[sumDimension] times to get weights for convolution and normalize through sumDimension */
        algorithmFPType divider = (algorithmFPType)1.0 / dataDims[sumDimension];

        for(size_t i = 0; i < dataDims[sumDimension]; i++)
        {
            for(size_t j = 0; j < nKernelElements; j++)
            {
                weightsArray[j + i * nKernelElements] = kernelArray[j] * divider;
            }
        }
    }
    else
    {
        /* Copy input kernel to convolution weights */
        for(size_t j = 0; j < nKernelElements; j++)
        {
            weightsArray[j] = kernelArray[j];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateCenteredDataAndSigma(const lcn::Parameter *parameter, Tensor *inputTensor)
{
    /* Create forward convolution algorithm and set algorithm parameters */
    if (this->_errors->size() > 0) { return; }

    SmartPtr<cpu> weightsBlock( nWeightsElements * sizeof(algorithmFPType));
    algorithmFPType *weightsArray = (algorithmFPType *)weightsBlock.get();
    if (!weightsArray)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    getConvolutionWeightsFromInputKernel(weightsArray, parameter);
    if (this->_errors->size() > 0) { return; }

    SmartPtr<cpu> biasesBlock( 1 * sizeof(algorithmFPType));
    algorithmFPType *biasesArray = (algorithmFPType *)biasesBlock.get();
    if (!biasesArray)  { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> weightsTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(weightsDims, weightsArray));
    if(weightsTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Fill null convolution biases */
    services::Collection<size_t> bDims(1);
    bDims[0] = nKernels;
    biasesArray[0] = (algorithmFPType)0;

    TensorOffsetLayout inputLayout = inputTensor->createDefaultSubtensorLayout();
    inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(*inputTensor, 0, 0, 0, nInputRows, inputLayout);
    inputArray = inputBlock.get(); /* already repacked array that has 1st dim as sumDimension, 2nd as firstDim and 3rd as secondDim */

    SharedPtr<Tensor> convInputTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(dataDims, const_cast<algorithmFPType *>(inputArray)));
    if(convInputTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> biasesTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(bDims, biasesArray));
    if(biasesTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    SharedPtr<Tensor> valueTensor  = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(sDims, const_cast<algorithmFPType *>(sigmaArray)));
    if(valueTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* step_1:  x_3 = conv(data) */
    convKernel.compute(convInputTensor.get(), weightsTensor.get(), biasesTensor.get(), &convParameter, valueTensor.get());

    size_t dataIndex = 0;
    size_t sigmaIndex = 0;
    for(size_t i = 0; i < dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < dataDims[sumDimension]; j++)
        {
            for(size_t k = 0; k < dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j * dataOffsetAfterDim + i * dataDims[sumDimension] * dataOffsetAfterDim;
                sigmaIndex = k + i * dataOffsetAfterDim;
                /* step_2:  x_4 = centeredData = data - step_1 */
                centeredDataArray[dataIndex] = inputArray[dataIndex] - sigmaArray[sigmaIndex];
            }
        }
    }
    inputBlock.release();

    /* step_3:  x_7 = pow(step_2, 2) */
    daal::internal::Math<algorithmFPType, cpu>::vPowx(nDataElements, centeredDataArray, (algorithmFPType)2, resultArray);

    convInputTensor = SharedPtr<Tensor>(new HomogenTensor<algorithmFPType>(dataDims, const_cast<algorithmFPType *>(resultArray)));
    if(convInputTensor == 0) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* step_4:  x_8 = conv(step_3) */
    convKernel.compute(convInputTensor.get(), weightsTensor.get(), biasesTensor.get(), &convParameter, valueTensor.get());

    /* step_5:  x_9 = sigma = sqrt(step_4) */
    daal::internal::Math<algorithmFPType, cpu>::vSqrt(nSigmaElements, sigmaArray, sigmaArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateCTensorAndGetMaxArray(Tensor *cTensor)
{
    const algorithmFPType one = 1.0;
    const algorithmFPType zero = 0;

    algorithmFPType divider = one / dataOffsetAfterDim;

    WriteSubtensor<algorithmFPType, cpu, Tensor> cBlock(*cTensor, 0, 0, 0, nCRows);
    cArray = cBlock.get();

    for(size_t i = 0; i < dataOffsetBeforeDim; i++)
    {
        cArray[i] = zero;
        for(size_t j = 0; j < dataOffsetAfterDim; j++)
        {
            /* step_6:  x_12 = c = mean(step_5) */
            cArray[i] += sigmaArray[j + i * dataOffsetAfterDim];
        }

        cArray[i] *= divider;

        for(size_t j = 0; j < dataOffsetAfterDim; j++)
        {
            /* step_7:  1/x_13 = 1 / max(step_5, step_6) */
            invMaxArray[j + i * dataOffsetAfterDim] = one / daal::internal::Math<algorithmFPType, cpu>::sMax(sigmaArray[j + i * dataOffsetAfterDim], cArray[i]);
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::calculateResult()
{
    size_t dataIndex = 0;
    size_t sigmaIndex = 0;
    for(size_t i = 0; i < dataOffsetBeforeDim; i++)
    {
        for(size_t j = 0; j < dataDims[sumDimension]; j++)
        {
            for(size_t k = 0; k < dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j * dataOffsetAfterDim + i * dataDims[sumDimension] * dataOffsetAfterDim;
                sigmaIndex = k + i * dataOffsetAfterDim;
                /* step_8: result = step_2 * step_7  */
                resultArray[dataIndex] = centeredDataArray[dataIndex] * invMaxArray[sigmaIndex];
            }
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::reset()
{
    convKernel.reset();
    dataDims.clear();
    kernelDims.clear();
    weightsDims.clear();
    sDims.clear();
}

} // internal
} // forward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
