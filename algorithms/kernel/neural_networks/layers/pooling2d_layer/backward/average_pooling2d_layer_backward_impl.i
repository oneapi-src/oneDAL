/* file: average_pooling2d_layer_backward_impl.i */
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
//  Implementation of backward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_memory.h"
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
namespace backward
{
namespace internal
{


template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::initialize(const services::Collection<size_t> &inDims,
        const services::Collection<size_t> &outDims)
{
    size_t dimension = inDims.size();

    inputSize = inputSizePtr.reset(dimension);
    inputStrides = inputStridesPtr.reset(dimension);
    outputSize = outputSizePtr.reset(dimension);
    outputStrides = outputStridesPtr.reset(dimension);
    DAAL_CHECK_MALLOC(inputSize && inputStrides && outputSize && outputStrides);

    inputSize    [0] = inDims[dimension - 1];
    inputStrides [0] = 1;
    outputSize   [0] = outDims[dimension - 1];
    outputStrides[0] = 1;

    for(size_t i = 1; i < dimension; i++)
    {
        inputSize    [i] = inDims[dimension - 1 - i];
        inputStrides [i] = inputStrides[i - 1] * inputSize[i - 1];
        outputSize   [i] = outDims[dimension - 1 - i];
        outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
    }

    ltUserInput  = xDnnLayout(dimension,  inputSize,  inputStrides); ON_ERR(ltUserInput.err);
    ltUserOutput = xDnnLayout(dimension, outputSize, outputStrides); ON_ERR(ltUserOutput.err);
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status PoolingKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputGradTensor, const pooling2d::Parameter &parameter,
        Tensor &gradTensor, const Tensor *dataTensor)
{
    const algorithmFPType zero = 0.0;

    const Collection<size_t> &inputGradDims = inputGradTensor.getDimensions();
    const Collection<size_t> &gradDims = gradTensor.getDimensions();

    MklTensor<algorithmFPType> *dataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(dataTensor));
    MklTensor<algorithmFPType> *inputGradMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(&inputGradTensor));
    MklTensor<algorithmFPType> *gradMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&gradTensor);

    if (dataMklTensor != 0)
    {
        dnnLayout_t inputLayout = (dnnLayout_t)dataMklTensor->getDnnLayout();
        dnnLayout_t inputGradLayout;
        dnnLayout_t workspaceLayout;
        dnnLayout_t resultLayout;
        dnnError_t err;

        algorithmFPType *avePoolRes[dnnResourceNumber] = {0};

        if (avePoolPrim == NULL)
        {
            const int inputOffset[2] = { (int)(-parameter.paddings.size[0]), (int)(-parameter.paddings.size[1]) };
            err = dnn::xPoolingCreateBackward(&avePoolPrim, dnnAlgorithmPoolingAvg, inputLayout,
                                              parameter.kernelSizes.size, parameter.strides.size, inputOffset, dnnBorderZeros);
            ON_ERR(err);
        }

        ReadSubtensor<algorithmFPType, cpu> inputGradBlock;
        LayoutConvertor<algorithmFPType, cpu> cvToInnerInputGrad;

        if (inputGradMklTensor != NULL)
        {
            err = dnn::xLayoutCreateFromPrimitive(&inputGradLayout, avePoolPrim, dnnResourceDiffDst); ON_ERR(err);
            inputGradMklTensor->setDnnLayout(inputGradLayout);
            avePoolRes[dnnResourceDiffDst] = inputGradMklTensor->getDnnArray();
        }
        else
        {
            err = dnn::xLayoutCreateFromPrimitive(&inputGradLayout, avePoolPrim, dnnResourceDiffDst); ON_ERR(err);

            inputGradBlock.set(const_cast<Tensor &>(inputGradTensor), 0, 0, 0, inputGradDims[0]);
            algorithmFPType *inputGradArray = const_cast<algorithmFPType *>(inputGradBlock.get());

            cvToInnerInputGrad.set(&inputGradArray, ltUserInput.get(), true, &avePoolRes[dnnResourceDiffDst], inputGradLayout, false); ON_ERR(cvToInnerInputGrad.err);
            cvToInnerInputGrad.convert(); ON_ERR(cvToInnerInputGrad.err);

            dnn::xLayoutDelete(inputGradLayout);
        }

        avePoolRes[dnnResourceWorkspace] = avePoolRes[dnnResourceDiffDst];

        if (gradMklTensor != NULL)
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, avePoolPrim, dnnResourceDiffSrc); ON_ERR(err);
            gradMklTensor->setDnnLayout(resultLayout);
            avePoolRes[dnnResourceDiffSrc] = gradMklTensor->getDnnArray();

            err = dnn::xExecute(avePoolPrim, (void **)avePoolRes); ON_ERR(err);
        }
        else
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, avePoolPrim, dnnResourceDiffSrc); ON_ERR(err);

            WriteOnlySubtensor<algorithmFPType, cpu> gradBlock(gradTensor, 0, 0, 0, gradDims[0]);
            algorithmFPType *gradArray = gradBlock.get();

            LayoutConvertor<algorithmFPType, cpu> cvFromInnerOutput(&avePoolRes[dnnResourceDiffSrc], resultLayout, false, &gradArray, ltUserOutput.get(), true); ON_ERR(cvFromInnerOutput.err);

            err = dnn::xExecute(avePoolPrim, (void **)avePoolRes); ON_ERR(err);

            cvFromInnerOutput.convert(); ON_ERR(cvFromInnerOutput.err);

            dnn::xLayoutDelete(resultLayout);
        }
    }
    else
    {
        ReadSubtensor<algorithmFPType, cpu> inputBlock(const_cast<Tensor&>(inputGradTensor), 0, 0, 0, inputGradDims[0]);
        DAAL_CHECK_BLOCK_STATUS(inputBlock);
        const algorithmFPType *inputGrad = inputBlock.get();

        WriteOnlySubtensor<algorithmFPType, cpu> gradBlock(gradTensor, 0, 0, 0, gradDims[0]);
        DAAL_CHECK_BLOCK_STATUS(gradBlock);
        algorithmFPType *grad = gradBlock.get();

        daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradBlock.getSize());

        pooling2d::internal::Parameter par(parameter.indices.size, parameter.paddings   .size,
                                           parameter.strides.size, parameter.kernelSizes.size,
                                           gradTensor, gradDims, inputGradDims);

        defaultCompute(par, inputGrad, NULL, grad);
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::defaultInnerLoop(
    const pooling2d::internal::Parameter &par,
    DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s,
    const algorithmFPType *inputGradPtr, const int *selectedPosPtr,
    algorithmFPType *grad)
{
    const algorithmFPType one = 1.0;
    const algorithmFPType gradMultiplier = one / (algorithmFPType)(par.firstKernelSize * par.secondKernelSize);
    for (DAAL_INT j = 0; j < par.offsetAfter; j++)
    {
        algorithmFPType inputValue = gradMultiplier * inputGradPtr[j];

        /*
         * Loops over the kernel
         */
        for (DAAL_INT fi = f; fi < f + par.firstKernelSize; fi++)
        {
            for (DAAL_INT si = s; si < s + par.secondKernelSize; si++)
            {
                DAAL_INT gradIndex = j + par.offsetAfter * (si + par.secondSize * (k + par.offsetBetween * (fi + par.firstSize * i)));
                bool paddingFlag = ((fi < 0) || (fi >= par.firstSize) || (si < 0) || (si >= par.secondSize));

                if (!paddingFlag)
                {
                    grad[gradIndex] += inputValue;
                }
            }
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
