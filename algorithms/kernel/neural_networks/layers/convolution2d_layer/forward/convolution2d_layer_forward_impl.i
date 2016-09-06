/* file: convolution2d_layer_forward_impl.i */
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
//  Implementation of convolution2d algorithm
//--
*/

#include "service_dnn.h"

#define ON_ERR(err) {                                                                                   \
    if ((err) != E_SUCCESS) {                                                                           \
        if((err) == E_MEMORY_ERROR) {this->_errors->add(services::ErrorMemoryAllocationFailed);return;} \
        this->_errors->add(services::ErrorConvolutionInternal);                                         \
        return;                                                                                         \
    }                                                                                                   \
}

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace convolution2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void Convolution2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *wTensor, Tensor *bTensor,
                                                                const convolution2d::Parameter *parameter, Tensor *resultTensor)
{
    dnnError_t err;
    typedef Dnn<algorithmFPType, cpu> dnn;

    const size_t dimension = 4;
    const size_t nGroups = parameter->nGroups;

    const services::Collection<size_t>& inDimsFull  = inputTensor->getDimensions();
    const services::Collection<size_t>& wDims       = wTensor->getDimensions();
    const services::Collection<size_t>& bDims       = bTensor->getDimensions();
    const services::Collection<size_t>& outDimsFull = resultTensor->getDimensions();
    services::Collection<size_t> inDims (dimension);
    services::Collection<size_t> outDims(dimension);

    const size_t dimsArray[4] = { 0, parameter->groupDimension, parameter->indices.dims[0], parameter->indices.dims[1] };
    TensorOffsetLayout targetInLayout = inputTensor->createDefaultSubtensorLayout();
    targetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensorEx(0, 0, 0, inDimsFull[0], readOnly, inputBlock, targetInLayout);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTensor->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bBlock;
    bTensor->getSubtensor(0, 0, 0, bDims[0], readOnly, bBlock);
    algorithmFPType *bArray = bBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, outDimsFull[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    inDims  = inDimsFull ;
    outDims = outDimsFull;

    const size_t bufferSize = 6 * dimension * sizeof(size_t);
    size_t *buffer = (size_t*)services::daal_malloc(bufferSize);
    if(!buffer) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}
    size_t *inputSize     = buffer + 0 * dimension;
    size_t *inputStrides  = buffer + 1 * dimension;
    size_t *outputSize    = buffer + 2 * dimension;
    size_t *outputStrides = buffer + 3 * dimension;
    size_t *filterSize    = buffer + 4 * dimension;
    size_t *filterStrides = buffer + 5 * dimension;

    size_t  biasSize   [1] = {parameter->nKernels};
    size_t  biasStrides[1] = {1};

    inputSize    [0] = inDims [dimension-1];
    inputStrides [0] = 1;
    outputSize   [0] = outDims[dimension-1];
    outputStrides[0] = 1;
    filterSize   [0] = wDims  [dimension-1];
    filterStrides[0] = 1;

    for(size_t i=1; i<dimension; i++)
    {
        inputSize    [i] = inDims [dimension-1-i];
        inputStrides [i] = inputStrides [i-1]*inputSize[i-1];
        outputSize   [i] = outDims[dimension-1-i];
        outputStrides[i] = outputStrides[i-1]*outputSize[i-1];
        filterSize   [i] = wDims  [dimension-1-i];
        filterStrides[i] = filterStrides[i-1]*filterSize[i-1];
    }

    size_t convolutionStride[2] = {  parameter->strides.size[1]  ,   parameter->strides.size[0]  };
    int    inputOffset      [2] = {-(parameter->paddings.size[1]), -(parameter->paddings.size[0])};

    dnnLayout_t ltUserInput=NULL, ltUserFilt=NULL, ltUserBias=NULL, ltUserOutput=NULL;

    err = dnn::xLayoutCreate(&ltUserInput,  dimension, inputSize,  inputStrides ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserFilt,   dimension, filterSize, filterStrides); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserBias,   1,         biasSize,   biasStrides  ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserOutput, dimension, outputSize, outputStrides); ON_ERR(err);

    dnnPrimitive_t convPrim = NULL;

    err = dnn::xConvolutionCreateForwardBias( &convPrim, dnnAlgorithmConvolutionDirect, nGroups, dimension, inputSize, outputSize,
        filterSize, convolutionStride, inputOffset, dnnBorderZeros);  ON_ERR(err);

    dnnLayout_t ltInnerInput = NULL, ltInnerFilt = NULL, ltInnerBias = NULL, ltInnerOutput = NULL;

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerInput,  convPrim, dnnResourceSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerFilt,   convPrim, dnnResourceFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerBias,   convPrim, dnnResourceBias  ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerOutput, convPrim, dnnResourceDst   ); ON_ERR(err);

    algorithmFPType* uOut;
    algorithmFPType* convRes[dnnResourceNumber] = {0};
    LayoutConvertor<algorithmFPType, cpu> cvToInnerInput(&inputArray, ltUserInput, true, &convRes[dnnResourceSrc   ], ltInnerInput, false); ON_ERR(cvToInnerInput.err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerFilt (&wArray    , ltUserFilt , true, &convRes[dnnResourceFilter], ltInnerFilt , false); ON_ERR(cvToInnerFilt .err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerBias (&bArray    , ltUserBias , true, &convRes[dnnResourceBias  ], ltInnerBias , false); ON_ERR(cvToInnerBias .err);

    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDst], ltInnerOutput); ON_ERR(err);

    LayoutConvertor<algorithmFPType, cpu> cvFromInnerOutput(&convRes[dnnResourceDst], ltInnerOutput, true, &uOut, ltUserOutput, false); ON_ERR(cvFromInnerOutput.err);

    cvToInnerInput.convert(); ON_ERR(cvToInnerInput.err);
    cvToInnerFilt .convert(); ON_ERR(cvToInnerFilt .err);
    cvToInnerBias .convert(); ON_ERR(cvToInnerBias .err);

    err = dnn::xExecute(convPrim, (void**)convRes); ON_ERR(err);

    cvFromInnerOutput.convert(); ON_ERR(cvFromInnerOutput.err);

    size_t size = resultBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        resultArray[i] = uOut[i];
    }

    dnn::xReleaseBuffer(convRes[dnnResourceDst]);

    dnn::xDelete(convPrim);

    dnn::xLayoutDelete(ltUserInput);
    dnn::xLayoutDelete(ltUserFilt);
    dnn::xLayoutDelete(ltUserBias);
    dnn::xLayoutDelete(ltUserOutput);
    dnn::xLayoutDelete(ltInnerInput);
    dnn::xLayoutDelete(ltInnerFilt);
    dnn::xLayoutDelete(ltInnerBias);
    dnn::xLayoutDelete(ltInnerOutput);

    services::daal_free(buffer);

    inputTensor->releaseSubtensor(inputBlock);
    wTensor->releaseSubtensor(wBlock);
    bTensor->releaseSubtensor(bBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // internal
} // forward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
