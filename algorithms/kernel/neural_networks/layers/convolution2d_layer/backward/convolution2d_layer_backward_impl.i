/* file: convolution2d_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of convolution2d algorithm
//--
*/

#include "service_dnn.h"
#include "service_dnn_internal.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

#include "service_mkl_tensor.h"

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
Status Convolution2dKernel<algorithmFPType, method, cpu>::initialize(bool resultFlag, bool wDerFlag, bool bDerFlag)
{
    _resultFlag = resultFlag;
    _wDerFlag   = wDerFlag;
    _bDerFlag   = bDerFlag;
    return Status();
}

template<CpuType cpu>
void fillDimensionArray(const services::Collection<size_t>& dims, size_t* sizes, size_t* strides)
{
    const size_t dimension = dims.size();

    sizes   [0] = dims[dimension-1];
    strides [0] = 1;

    for(size_t i=1; i<dimension; i++)
    {
        sizes       [i] = dims[dimension-1-i];
        strides     [i] = strides[i-1]*sizes[i-1];
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status Convolution2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor,
    const convolution2d::Parameter &parameter, Tensor *wDerTensor, Tensor *bDerTensor, Tensor *resultTensor)
{
    Status s;

    MklTensor<algorithmFPType> *xMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(xTensor);
    MklTensor<algorithmFPType> *inGradMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inGradTensor);
    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(resultTensor);

    dnnError_t err;
    typedef Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;
    typedef daal::internal::DnnBuffer<algorithmFPType, cpu> xDnnBuffer;

    const size_t dimension = 4;
    const size_t nGroups = parameter.nGroups;
    const size_t filtDimension = dimension + (nGroups != 1);

    const services::Collection<size_t>& gDimsFull = inGradTensor->getDimensions();
    services::Collection<size_t> xDims;
    if(resultTensor)
        xDims = resultTensor->getDimensions();
    else
        xDims = xTensor->getDimensions();
    const services::Collection<size_t>& wDims = wTensor->getDimensions();
    services::Collection<size_t> bDims;
    bDims << parameter.nKernels;

    const size_t dimsArray[4] = { 0, parameter.groupDimension, parameter.indices.dims[0], parameter.indices.dims[1] };

    services::Collection<size_t> gDims(dimension);
    gDims = gDimsFull;
    const services::Collection<size_t>& xDimsFull = xDims;

    size_t xSize        [dimension];
    size_t xStrides     [dimension];
    size_t filterSize   [dimension + 1];
    size_t filterStrides[dimension + 1];
    size_t gradSize     [dimension];
    size_t gradStrides  [dimension];

    size_t  biasSize   [1] = {parameter.nKernels};
    size_t  biasStrides[1] = {1};

    fillDimensionArray<cpu>(gDims, gradSize, gradStrides);

    size_t convolutionStride[2] = { parameter.strides.size[1] ,  parameter.strides.size[0] };
    int    xOffset          [2] = {-parameter.paddings.size[1], -parameter.paddings.size[0]};

    dnnPrimitive_t auxPrim  = NULL;

    xDnnLayout ltUserGrad(dimension, gradSize,   gradStrides  ); ON_ERR(ltUserGrad.err);

    xDnnLayout ltUserX   ;
    xDnnLayout ltUserFilt;
    xDnnLayout ltUserBias;

    if(_resultFlag && parameter.propagateGradient)
    {
        if(!ltUserX.get())
        {
            fillDimensionArray<cpu>(xDims, xSize, xStrides);
            ltUserX = xDnnLayout(dimension, xSize, xStrides); ON_ERR(ltUserX.err);
        }

        if(!ltUserFilt.get())
        {
            fillDimensionArray<cpu>(wDims, filterSize, filterStrides);
            ltUserFilt = xDnnLayout(filtDimension, filterSize, filterStrides); ON_ERR(ltUserFilt.err);
        }
        if (convGrad == NULL)
        {
            err = dnn::xConvolutionCreateBackwardData  (&convGrad, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                        filterSize, convolutionStride, xOffset, dnnBorderZeros); ON_ERR(err);
        }
        if (!auxPrim) auxPrim = convGrad;
    }
    else
    {
        if (convGrad != NULL)
        {
            dnn::xDelete(convGrad);
            convGrad = NULL;
        }
    }

    if(_wDerFlag)
    {
        if(!ltUserX.get())
        {
            fillDimensionArray<cpu>(xDims, xSize, xStrides);
            ltUserX = xDnnLayout(dimension, xSize, xStrides); ON_ERR(ltUserX.err);
        }

        if(!ltUserFilt.get())
        {
            fillDimensionArray<cpu>(wDims, filterSize, filterStrides);
            ltUserFilt = xDnnLayout(filtDimension, filterSize, filterStrides); ON_ERR(ltUserFilt.err);
        }

        if (convFilt == NULL)
        {
            err = dnn::xConvolutionCreateBackwardFilter(&convFilt, dnnAlgorithmConvolutionDirect, nGroups, dimension, xSize, gradSize,
                                                        filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);

        }
        if (!auxPrim) auxPrim = convFilt;
    }
    else
    {
        if (convFilt != NULL)
        {
            dnn::xDelete(convFilt);
            convFilt = NULL;
        }
    }

    if(_bDerFlag)
    {
        if(!ltUserBias.get())
        {
            ltUserBias = xDnnLayout(1, biasSize, biasStrides); ON_ERR(ltUserBias.err);
        }

        if (convBias == NULL)
        {
            err = dnn::xConvolutionCreateBackwardBias  (&convBias, dnnAlgorithmConvolutionDirect, nGroups, dimension, gradSize);  ON_ERR(err);
        }
        if (!auxPrim) auxPrim = convBias;
    }
    else
    {
        if (convBias != NULL)
        {
            dnn::xDelete(convBias);
            convBias = NULL;
        }
    }

    algorithmFPType* convRes[dnnResourceNumber] = {0};

    dnnLayout_t inGradLayout;
    err = dnn::xLayoutCreateFromPrimitive(&inGradLayout, auxPrim, dnnResourceDiffDst); ON_ERR(err);

    ReadSubtensor<algorithmFPType, cpu> inGradBlock;
    LayoutConvertor<algorithmFPType, cpu> cvToInnerGrad;

    if (inGradMklTensor != 0)
    {
        inGradMklTensor->setDnnLayout(inGradLayout);
        convRes[dnnResourceDiffDst] = inGradMklTensor->getDnnArray();
    }
    else
    {
        TensorOffsetLayout gTargetInLayout = inGradTensor->createDefaultSubtensorLayout();
        DAAL_CHECK_STATUS(s, gTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) ) );
        inGradBlock.set(inGradTensor, 0, 0, 0, gDimsFull[0], gTargetInLayout);
        DAAL_CHECK_BLOCK_STATUS(inGradBlock);
        algorithmFPType *inGradArray = const_cast<algorithmFPType*>(inGradBlock.get());

        cvToInnerGrad.set(&inGradArray, ltUserGrad.get(), true, &convRes[dnnResourceDiffDst], inGradLayout, false); ON_ERR(cvToInnerGrad.err);
        cvToInnerGrad.convert(); ON_ERR(cvToInnerGrad.err);

        dnn::xLayoutDelete(inGradLayout);
    }

    if(convGrad)
    {
        ReadSubtensor<algorithmFPType, cpu> wBlock(wTensor, 0, 0, 0, wDims[0]);
        DAAL_CHECK_BLOCK_STATUS(wBlock);
        algorithmFPType *wArray = const_cast<algorithmFPType*>(wBlock.get());
        xDnnLayout ltInnerFilt (convGrad, dnnResourceFilter); ON_ERR(ltInnerFilt.err);

        LayoutConvertor<algorithmFPType, cpu> cvToInnerFilt (&wArray, ltUserFilt.get(), true, &convRes[dnnResourceFilter ], ltInnerFilt .get(), false); ON_ERR(cvToInnerFilt .err);
        cvToInnerFilt .convert(); ON_ERR(cvToInnerFilt .err);

        dnnLayout_t resultLayout;
        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, convGrad, dnnResourceDiffSrc); ON_ERR(err);

        WriteOnlySubtensor<algorithmFPType, cpu> resultBlock;
        LayoutConvertor<algorithmFPType, cpu> cvFromInnerBack;

        if (resultMklTensor != 0)
        {
            resultMklTensor->setDnnLayout(resultLayout);
            convRes[dnnResourceDiffSrc] = resultMklTensor->getDnnArray();

            err = dnn::xExecute(convGrad, (void**)convRes); ON_ERR(err);
        }
        else
        {
            resultBlock.set(resultTensor, 0, 0, 0, xDimsFull[0]);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);
            algorithmFPType *resultArray = resultBlock.get();

            cvFromInnerBack.set(&convRes[dnnResourceDiffSrc], resultLayout, false, &resultArray, ltUserX.get(), true); ON_ERR(cvFromInnerBack.err);

            err = dnn::xExecute(convGrad, (void**)convRes); ON_ERR(err);

            cvFromInnerBack.convert(); ON_ERR(cvFromInnerBack.err);
            dnn::xLayoutDelete(resultLayout);
        }
    }

    algorithmFPType *derFilt=0, *derBias=0;

    if(convFilt)
    {
        WriteOnlySubtensor<algorithmFPType, cpu> wDerBlock(wDerTensor, 0, 0, 0, wDims[0]);
        DAAL_CHECK_BLOCK_STATUS(wDerBlock);
        algorithmFPType *wDerArray = wDerBlock.get();

        dnnLayout_t inputLayout;
        err = dnn::xLayoutCreateFromPrimitive(&inputLayout, convFilt, dnnResourceSrc); ON_ERR(err);

        ReadSubtensor<algorithmFPType, cpu> xBlock;
        LayoutConvertor<algorithmFPType, cpu> cvToInnerInput;

        if (xMklTensor != 0)
        {
            xMklTensor->setDnnLayout(inputLayout);
            convRes[dnnResourceSrc] = xMklTensor->getDnnArray();
        }
        else
        {
            TensorOffsetLayout xTargetInLayout = xTensor->createDefaultSubtensorLayout();
            xTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );
            xBlock.set(xTensor, 0, 0, 0, xDimsFull[0], xTargetInLayout);
            algorithmFPType *xArray = const_cast<algorithmFPType*>(xBlock.get());

            cvToInnerInput.set(&xArray,      ltUserX   .get(), true, &convRes[dnnResourceSrc], inputLayout, false); ON_ERR(cvToInnerInput.err);
            cvToInnerInput.convert(); ON_ERR(cvToInnerInput.err);

            dnn::xLayoutDelete(inputLayout);
        }

        algorithmFPType batchSize = (algorithmFPType)xDimsFull[0];
        algorithmFPType invBatchSize = 1.0 / batchSize;

        xDnnLayout ltInnerDerFilt(convFilt, dnnResourceDiffFilter); ON_ERR(ltInnerDerFilt.err);
        xDnnBuffer dnnResourceDiffFilterBuffer(ltInnerDerFilt.get()); ON_ERR(dnnResourceDiffFilterBuffer.err);
        convRes[dnnResourceDiffFilter] = dnnResourceDiffFilterBuffer.get();

        LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerFilt(&convRes[dnnResourceDiffFilter], ltInnerDerFilt.get(), true, &derFilt , ltUserFilt.get(), false); ON_ERR(cvFromInnerDerFilt.err);
        err = dnn::xExecute(convFilt, (void**)convRes); ON_ERR(err);
        cvFromInnerDerFilt.convert(); ON_ERR(cvFromInnerDerFilt.err);

        size_t size = wDerBlock.getSize();
        for(size_t i=0; i<size; i++)
        {
            wDerArray[i] = invBatchSize * derFilt[i];
        }
    }

    if(convBias)
    {
        WriteOnlySubtensor<algorithmFPType, cpu> bDerBlock(bDerTensor, 0, 0, 0, bDims[0]);
        algorithmFPType *bDerArray = bDerBlock.get();

        algorithmFPType batchSize = (algorithmFPType)xDimsFull[0];
        algorithmFPType invBatchSize = 1.0 / batchSize;

        xDnnLayout ltInnerDerBias(convBias, dnnResourceDiffBias  ); ON_ERR(ltInnerDerBias.err);
        xDnnBuffer dnnResourceDiffBiasBuffer  (ltInnerDerBias.get()); ON_ERR(dnnResourceDiffBiasBuffer.err);
        convRes[dnnResourceDiffBias]   = dnnResourceDiffBiasBuffer.get();

        LayoutConvertor<algorithmFPType, cpu> cvFromInnerDerBias(&convRes[dnnResourceDiffBias  ], ltInnerDerBias.get(), true, &derBias , ltUserBias.get(), false); ON_ERR(cvFromInnerDerBias.err);
        err = dnn::xExecute(convBias, (void**)convRes); ON_ERR(err);
        cvFromInnerDerBias.convert(); ON_ERR(cvFromInnerDerBias.err);

        size_t size = bDerBlock.getSize();
        for(size_t i=0; i<size; i++)
        {
            bDerArray[i] = invBatchSize * derBias[i];
        }
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status Convolution2dKernel<algorithmFPType, method, cpu>::reset()
{
    if(convGrad != NULL)
    {
        dnn::xDelete(convGrad);
        convGrad = NULL;
    }
    if(convFilt != NULL)
    {
        dnn::xDelete(convFilt);
        convFilt = NULL;
    }
    if(convBias != NULL)
    {
        dnn::xDelete(convBias);
        convBias = NULL;
    }
    return services::Status();
}

} // internal
} // backward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
