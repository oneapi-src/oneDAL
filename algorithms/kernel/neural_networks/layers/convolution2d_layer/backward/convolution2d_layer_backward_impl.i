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
#include "service_dnn_internal.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

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
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;
    typedef daal::internal::DnnBuffer<algorithmFPType, cpu> xDnnBuffer;

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

    ReadSubtensor<algorithmFPType, cpu> inGradBlock(inGradTensor, 0, 0, 0, gDimsFull[0], gTargetInLayout);
    algorithmFPType *inGradArray = const_cast<algorithmFPType*>(inGradBlock.get());

    ReadSubtensor<algorithmFPType, cpu> xBlock(xTensor, 0, 0, 0, xDimsFull[0], xTargetInLayout);
    algorithmFPType *xArray = const_cast<algorithmFPType*>(xBlock.get());

    ReadSubtensor<algorithmFPType, cpu> wBlock(wTensor, 0, 0, 0, wDims[0]);
    algorithmFPType *wArray = const_cast<algorithmFPType*>(wBlock.get());

    WriteOnlySubtensor<algorithmFPType, cpu> wDerBlock(wDerTensor, 0, 0, 0, wDims[0]);
    algorithmFPType *wDerArray = wDerBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> bDerBlock(bDerTensor, 0, 0, 0, bDims[0]);
    algorithmFPType *bDerArray = bDerBlock.get();

    gDims = gDimsFull;
    xDims = xDimsFull;

    daal::internal::TSmartPtr<size_t, cpu> bufferPtr(6 * dimension);
    size_t *buffer = bufferPtr.get();
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

    xDnnLayout ltUserX   (dimension, xSize,      xStrides     ); ON_ERR(ltUserX.err);
    xDnnLayout ltUserFilt(dimension, filterSize, filterStrides); ON_ERR(ltUserFilt.err);
    xDnnLayout ltUserBias(1,         biasSize,   biasStrides  ); ON_ERR(ltUserBias.err);
    xDnnLayout ltUserGrad(dimension, gradSize,   gradStrides  ); ON_ERR(ltUserGrad.err);

    dnnPrimitive_t convFwd = NULL, convGrad = NULL, convFilt = NULL, convBias = NULL;

    err = dnn::xConvolutionCreateForwardBias   (&convFwd,  dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardFilter(&convFilt, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardBias  (&convBias, dnnAlgorithmConvolutionDirect, nGroups, dimension, gradSize);  ON_ERR(err);

    xDnnLayout ltInnerInput(convFwd, dnnResourceSrc   ); ON_ERR(ltInnerInput.err);
    xDnnLayout ltInnerFilt (convFwd, dnnResourceFilter); ON_ERR(ltInnerFilt.err);
    xDnnLayout ltInnerGrad (convFwd, dnnResourceDst   ); ON_ERR(ltInnerGrad.err);

    xDnnLayout ltInnerDerFilt(convFilt, dnnResourceDiffFilter); ON_ERR(ltInnerDerFilt.err);
    xDnnLayout ltInnerDerBias(convBias, dnnResourceDiffBias  ); ON_ERR(ltInnerDerBias.err);

    algorithmFPType* convRes[dnnResourceNumber] = {0};
    LayoutConvertor<algorithmFPType, cpu> cvToInnerInput(&xArray,      ltUserX   .get(), true, &convRes[dnnResourceSrc    ], ltInnerInput.get(), false); ON_ERR(cvToInnerInput.err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerFilt (&wArray,      ltUserFilt.get(), true, &convRes[dnnResourceFilter ], ltInnerFilt .get(), false); ON_ERR(cvToInnerFilt .err);
    LayoutConvertor<algorithmFPType, cpu> cvToInnerGrad (&inGradArray, ltUserGrad.get(), true, &convRes[dnnResourceDiffDst], ltInnerGrad .get(), false); ON_ERR(cvToInnerGrad .err);

    cvToInnerInput.convert(); ON_ERR(cvToInnerInput.err);
    cvToInnerFilt .convert(); ON_ERR(cvToInnerFilt .err);
    cvToInnerGrad .convert(); ON_ERR(cvToInnerGrad .err);

    xDnnBuffer dnnResourceDiffFilterBuffer(ltInnerDerFilt.get()); ON_ERR(dnnResourceDiffFilterBuffer.err);
    xDnnBuffer dnnResourceDiffBiasBuffer  (ltInnerDerBias.get()); ON_ERR(dnnResourceDiffBiasBuffer.err);
    convRes[dnnResourceDiffFilter] = dnnResourceDiffFilterBuffer.get();
    convRes[dnnResourceDiffBias]   = dnnResourceDiffBiasBuffer.get();

    algorithmFPType *derFilt=0, *derBias=0;
    LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerFilt(&convRes[dnnResourceDiffFilter], ltInnerDerFilt.get(), true, &derFilt , ltUserFilt.get(), false); ON_ERR(cvFromInnerDerFilt.err);
    LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerBias(&convRes[dnnResourceDiffBias  ], ltInnerDerBias.get(), true, &derBias , ltUserBias.get(), false); ON_ERR(cvFromInnerDerBias.err);

    err = dnn::xExecute(convFilt, (void**)convRes); ON_ERR(err);
    err = dnn::xExecute(convBias, (void**)convRes); ON_ERR(err);

    cvFromInnerDerFilt.convert(); ON_ERR(cvFromInnerDerFilt.err);
    cvFromInnerDerBias.convert(); ON_ERR(cvFromInnerDerBias.err);

    if (parameter->propagateGradient)
    {
        dnnPrimitive_t convGrad = NULL;

        err = dnn::xConvolutionCreateBackwardData  (&convGrad, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                    filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);

        xDnnLayout ltInnerBack(convGrad, dnnResourceDiffSrc); ON_ERR(ltInnerBack.err);

        xDnnBuffer dnnResourceDiffSrcBuffer(ltInnerBack.get()); ON_ERR(dnnResourceDiffSrcBuffer.err);
        convRes[dnnResourceDiffSrc] = dnnResourceDiffSrcBuffer.get();

        WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, xDimsFull[0]);
        algorithmFPType *resultArray = resultBlock.get();

        LayoutConvertor<algorithmFPType, cpu> cvFromInnerBack(&convRes[dnnResourceDiffSrc], ltInnerBack.get(), false, &resultArray, ltUserX.get(), true); ON_ERR(cvFromInnerBack.err);

        err = dnn::xExecute(convGrad, (void**)convRes); ON_ERR(err);

        cvFromInnerBack.convert(); ON_ERR(cvFromInnerBack.err);

        dnn::xDelete(convGrad);
    }

    algorithmFPType batchSize = (algorithmFPType)xDimsFull[0];
    algorithmFPType invBatchSize = 1.0 / batchSize;

    size_t size = wDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        wDerArray[i] = invBatchSize * derFilt[i];
    }

    size = bDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        bDerArray[i] = invBatchSize * derBias[i];
    }

    dnn::xDelete(convFwd );
    dnn::xDelete(convFilt);
    dnn::xDelete(convBias);
}

} // internal
} // backward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
