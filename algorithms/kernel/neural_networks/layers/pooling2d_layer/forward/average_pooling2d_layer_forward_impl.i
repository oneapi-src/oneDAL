/* file: average_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_data_utils.h"
#include "service_blas.h"
#include "service_tensor.h"

#include "service_mkl_tensor.h"

using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::initialize(const services::Collection<size_t> &inDims,
        const services::Collection<size_t> &outDims)
{
    size_t dimension = inDims.size();

    outputSize = outputSizePtr.reset(dimension);
    outputStrides = outputStridesPtr.reset(dimension);
    DAAL_CHECK_MALLOC(outputSize && outputStrides);

    outputSize   [0] = outDims[dimension - 1];
    outputStrides[0] = 1;

    for(size_t i = 1; i < dimension; i++)
    {
        outputSize   [i] = outDims[dimension - 1 - i];
        outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
    }

    ltUserOutput = xDnnLayout(dimension, outputSize, outputStrides); ON_ERR(ltUserOutput.err);
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &dataTensor, const average_pooling2d::Parameter &parameter, Tensor &valueTensor)
{
    const Collection<size_t> &dims = dataTensor.getDimensions();
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    MklTensor<algorithmFPType> *dataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor*>(&dataTensor));
    MklTensor<algorithmFPType> *valueMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&valueTensor);

    if (dataMklTensor != NULL)
    {
        algorithmFPType *avePoolRes[dnnResourceNumber] = {0};

        dnnLayout_t inputLayout;
        dnnLayout_t workspaceLayout;
        dnnLayout_t resultLayout;

        inputLayout = (dnnLayout_t)dataMklTensor->getDnnLayout();
        avePoolRes[dnnResourceSrc] = dataMklTensor->getDnnArray();

        dnnError_t err;

        if (avePoolPrim == NULL)
        {
            const int inputOffset[2] = { (int)(-parameter.paddings.size[0]), (int)(-parameter.paddings.size[1]) };
            err = dnn::xPoolingCreateForward(&avePoolPrim, dnnAlgorithmPoolingAvg, inputLayout,
                                             parameter.kernelSizes.size, parameter.strides.size, inputOffset, dnnBorderZeros);
            ON_ERR(err);
        }

        if (valueMklTensor != NULL)
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, avePoolPrim, dnnResourceDst); ON_ERR(err);
            valueMklTensor->setDnnLayout(resultLayout);
            avePoolRes[dnnResourceDst] = valueMklTensor->getDnnArray();
            avePoolRes[dnnResourceWorkspace] = avePoolRes[dnnResourceDst];

            err = dnn::xExecute(avePoolPrim, (void **)avePoolRes); ON_ERR(err);
        }
        else
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, avePoolPrim, dnnResourceDst); ON_ERR(err);

            WriteOnlySubtensor<algorithmFPType, cpu> valueBlock(valueTensor, 0, 0, 0, valueDims[0]);
            algorithmFPType *valueArray = valueBlock.get();

            LayoutConvertor<algorithmFPType, cpu> cvFromInnerOutput(&avePoolRes[dnnResourceDst], resultLayout, false, &valueArray, ltUserOutput.get(), true); ON_ERR(cvFromInnerOutput.err);
            avePoolRes[dnnResourceWorkspace] = avePoolRes[dnnResourceDst];

            err = dnn::xExecute(avePoolPrim, (void **)avePoolRes); ON_ERR(err);

            cvFromInnerOutput.convert(); ON_ERR(cvFromInnerOutput.err);

            dnn::xLayoutDelete(resultLayout);
        }
    }
    else
    {
        const algorithmFPType zero = 0.0;
        const algorithmFPType one = 1.0;

        ReadSubtensor<algorithmFPType, cpu> dataBlock(const_cast<Tensor&>(dataTensor), 0, 0, 0, dims[0]);
        DAAL_CHECK_BLOCK_STATUS(dataBlock);
        const algorithmFPType *data = dataBlock.get();

        WriteOnlySubtensor<algorithmFPType, cpu> valueBlock(valueTensor, 0, 0, 0, valueDims[0]);
        DAAL_CHECK_BLOCK_STATUS(valueBlock);
        algorithmFPType *value = valueBlock.get();

        pooling2d::internal::Parameter par(parameter.indices.size, parameter.paddings.size,
                                           parameter.strides.size, parameter.kernelSizes.size,
                                           dataTensor, dims, valueDims);

        defaultCompute(par, data, value);
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(const pooling2d::internal::Parameter &par,
        DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
        const algorithmFPType *data, algorithmFPType *valuePtr)
{
    const algorithmFPType zero = 0.0;
    algorithmFPType average = zero;

    /*
     * Loops over the kernel
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
            const DAAL_INT dataIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));
            const bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));
            const algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

            average += dataValue;
        }
    }
    valuePtr[j] = average / (algorithmFPType)(par.firstKernelSize * par.secondKernelSize);
}

} // namespace internal
} // namespace forward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
