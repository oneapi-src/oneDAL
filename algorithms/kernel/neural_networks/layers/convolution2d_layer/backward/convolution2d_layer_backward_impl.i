/* file: convolution2d_layer_backward_impl.i */
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

#define ON_ERR(err) { \
    if ((err) != E_SUCCESS) { \
        if((err) == E_MEMORY_ERROR) {this->_errors->add(services::ErrorMemoryAllocationFailed);return;} \
        this->_errors->add(services::ErrorConvolutionInternal);\
        return; \
    } \
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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void Convolution2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor,
    const convolution2d::Parameter *parameter, Tensor *wDerTensor, Tensor *bDerTensor, Tensor *resultTensor)
{
    dnnError_t err;
    typedef Dnn<algorithmFPType, cpu> dnn;

    const size_t dimension = 4;
    const size_t nGroups = parameter->nGroups;

    const services::Collection<size_t>& gDimsFull = inGradTensor->getDimensions();
    const services::Collection<size_t>& xDimsFull = xTensor->getDimensions();
    const services::Collection<size_t>& wDims = wDerTensor->getDimensions();
    const services::Collection<size_t>& bDims = bDerTensor->getDimensions();
    services::Collection<size_t> gDims(dimension);
    services::Collection<size_t> xDims(dimension);

    const size_t dimsArray[4] = { 0, parameter->groupDimension, parameter->indices.dims[0], parameter->indices.dims[1] };
    TensorOffsetLayout gTargetInLayout = inGradTensor->createDefaultSubtensorLayout();
    TensorOffsetLayout xTargetInLayout = xTensor     ->createDefaultSubtensorLayout();
    gTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );
    xTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );

    SubtensorDescriptor<algorithmFPType> inGradBlock;
    inGradTensor->getSubtensorEx(0, 0, 0, gDimsFull[0], readOnly, inGradBlock, gTargetInLayout);
    algorithmFPType *inGradArray = inGradBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> xBlock;
    xTensor->getSubtensorEx(0, 0, 0, xDimsFull[0], readOnly, xBlock, xTargetInLayout);
    algorithmFPType *xArray = xBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTensor->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wDerBlock;
    wDerTensor->getSubtensor(0, 0, 0, wDims[0], writeOnly, wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bDerBlock;
    bDerTensor->getSubtensor(0, 0, 0, bDims[0], writeOnly, bDerBlock);
    algorithmFPType *bDerArray = bDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, xDimsFull[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    gDims = gDimsFull;
    xDims = xDimsFull;

    const size_t bufferSize = 6 * dimension * sizeof(size_t);
    size_t *buffer = (size_t*)services::daal_malloc(bufferSize);
    if(!buffer) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}
    size_t *xSize         = buffer + 0 * dimension;
    size_t *xStrides      = buffer + 1 * dimension;
    size_t *gradSize      = buffer + 2 * dimension;
    size_t *gradStrides   = buffer + 3 * dimension;
    size_t *filterSize    = buffer + 4 * dimension;
    size_t *filterStrides = buffer + 5 * dimension;

    size_t  biasSize   [1] = {parameter->nKernels};
    size_t  biasStrides[1] = {1};

    xSize        [0] = xDims[dimension-1];
    xStrides     [0] = 1;
    gradSize     [0] = gDims[dimension-1];
    gradStrides  [0] = 1;
    filterSize   [0] = wDims[dimension-1];
    filterStrides[0] = 1;

    for(size_t i=1; i<dimension; i++)
    {
        xSize        [i] = xDims[dimension-1-i];
        xStrides     [i] = xStrides[i-1]*xSize[i-1];
        gradSize     [i] = gDims[dimension-1-i];
        gradStrides  [i] = gradStrides[i-1]*gradSize[i-1];
        filterSize   [i] = wDims[dimension-1-i];
        filterStrides[i] = filterStrides[i-1]*filterSize[i-1];
    }

    size_t convolutionStride[2] = {parameter->strides.size[1],  parameter->strides.size[0]};
    int    xOffset          [2] = {-(int)(parameter->paddings.size[1]), -(int)(parameter->paddings.size[0])};

    dnnLayout_t ltUserX=NULL, ltUserFilt=NULL, ltUserBias=NULL, ltUserGrad=NULL;

    err = dnn::xLayoutCreate(&ltUserX,      dimension, xSize,      xStrides     ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserFilt,   dimension, filterSize, filterStrides); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserBias,   1,         biasSize,   biasStrides  ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserGrad,   dimension, gradSize,   gradStrides  ); ON_ERR(err);

    dnnPrimitive_t convFwd = NULL, convGrad = NULL, convFilt = NULL, convBias = NULL;

    err = dnn::xConvolutionCreateForwardBias   ( &convFwd,  dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardData  ( &convGrad, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardFilter( &convFilt, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardBias  ( &convBias, dnnAlgorithmConvolutionDirect, nGroups, dimension, gradSize);  ON_ERR(err);

    dnnLayout_t ltInnerInput = NULL, ltInnerFilt = NULL, ltInnerGrad = NULL;
    dnnLayout_t ltInnerBack  = NULL, ltInnerDerFilt = NULL, ltInnerDerBias = NULL;

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerInput, convFwd, dnnResourceSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerFilt,  convFwd, dnnResourceFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerGrad,  convFwd, dnnResourceDst   ); ON_ERR(err);

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerBack,    convGrad, dnnResourceDiffSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerDerFilt, convFilt, dnnResourceDiffFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerDerBias, convBias, dnnResourceDiffBias  ); ON_ERR(err);

    algorithmFPType* convRes[dnnResourceNumber] = {0};
    LayoutConvertor<algorithmFPType, cpu> cvToInnerInput(&xArray     , ltUserX   , true, &convRes[dnnResourceSrc    ], ltInnerInput, false); ON_ERR(cvToInnerInput.err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerFilt (&wArray     , ltUserFilt, true, &convRes[dnnResourceFilter ], ltInnerFilt , false); ON_ERR(cvToInnerFilt .err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerGrad (&inGradArray, ltUserGrad, true, &convRes[dnnResourceDiffDst], ltInnerGrad , false); ON_ERR(cvToInnerGrad .err);

    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffSrc   ], ltInnerBack   ); ON_ERR(err);
    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffFilter], ltInnerDerFilt); ON_ERR(err);
    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffBias  ], ltInnerDerBias); ON_ERR(err);

    algorithmFPType *backData=0, *derFilt=0, *derBias=0;
    LayoutConvertor<algorithmFPType, cpu> cvFromInnerBack   (&convRes[dnnResourceDiffSrc   ], ltInnerBack   , true, &backData, ltUserX   , false); ON_ERR(cvFromInnerBack   .err);
    LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerFilt(&convRes[dnnResourceDiffFilter], ltInnerDerFilt, true, &derFilt , ltUserFilt, false); ON_ERR(cvFromInnerDerFilt.err);
    LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerBias(&convRes[dnnResourceDiffBias  ], ltInnerDerBias, true, &derBias , ltUserBias, false); ON_ERR(cvFromInnerDerBias.err);

    cvToInnerInput.convert(); ON_ERR(cvToInnerInput.err);
    cvToInnerFilt .convert(); ON_ERR(cvToInnerFilt .err);
    cvToInnerGrad .convert(); ON_ERR(cvToInnerGrad .err);

    err = dnn::xExecute(convGrad, (void**)convRes); ON_ERR(err);
    err = dnn::xExecute(convFilt, (void**)convRes); ON_ERR(err);
    err = dnn::xExecute(convBias, (void**)convRes); ON_ERR(err);

    cvFromInnerBack   .convert(); ON_ERR(cvFromInnerBack   .err);
    cvFromInnerDerFilt.convert(); ON_ERR(cvFromInnerDerFilt.err);
    cvFromInnerDerBias.convert(); ON_ERR(cvFromInnerDerBias.err);

    size_t size = resultBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        resultArray[i] = backData[i];
    }

    algorithmFPType batchSize = (algorithmFPType)xDimsFull[0];
    algorithmFPType invBatchSize = 1.0 / batchSize;

    size = wDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        wDerArray[i] = invBatchSize * derFilt[i];
    }

    size = bDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        bDerArray[i] = invBatchSize * derBias[i];
    }

    dnn::xReleaseBuffer(convRes[dnnResourceDiffSrc   ]);
    dnn::xReleaseBuffer(convRes[dnnResourceDiffFilter]);
    dnn::xReleaseBuffer(convRes[dnnResourceDiffBias  ]);

    dnn::xDelete(convFwd );
    dnn::xDelete(convGrad);
    dnn::xDelete(convFilt);
    dnn::xDelete(convBias);

    dnn::xLayoutDelete(ltUserX);
    dnn::xLayoutDelete(ltUserFilt);
    dnn::xLayoutDelete(ltUserBias);
    dnn::xLayoutDelete(ltUserGrad);

    dnn::xLayoutDelete(ltInnerInput);
    dnn::xLayoutDelete(ltInnerFilt);
    dnn::xLayoutDelete(ltInnerGrad);

    dnn::xLayoutDelete(ltInnerBack);
    dnn::xLayoutDelete(ltInnerDerFilt);
    dnn::xLayoutDelete(ltInnerDerBias);

    services::daal_free(buffer);

    inGradTensor->releaseSubtensor(inGradBlock);
    xTensor->releaseSubtensor(xBlock);
    wTensor->releaseSubtensor(wBlock);
    wDerTensor->releaseSubtensor(wDerBlock);
    bDerTensor->releaseSubtensor(bDerBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // internal
} // backward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
