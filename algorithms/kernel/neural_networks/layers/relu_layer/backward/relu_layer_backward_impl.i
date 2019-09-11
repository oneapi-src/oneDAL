/* file: relu_layer_backward_impl.i */
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
//  Implementation of relu algorithm
//--
*/

#ifndef __RELU_LAYER_BACKWARD_IMPL_I__
#define __RELU_LAYER_BACKWARD_IMPL_I__

#include "service_mkl_tensor.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace relu
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status ReLUKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputGradientTensor, const Tensor &forwardDataTensor, Tensor &resultTensor)
{
    MklTensor<algorithmFPType> *inputGradientMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(&inputGradientTensor));
    MklTensor<algorithmFPType> *forwardDataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(&forwardDataTensor));
    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&resultTensor);

    Status s;

    if (inputGradientMklTensor != 0 && forwardDataMklTensor != 0 && resultMklTensor != 0)
    {
        dnnLayout_t inputLayout = (dnnLayout_t)forwardDataMklTensor->getDnnLayout();
        dnnLayout_t inputGradLayout = (dnnLayout_t)inputGradientMklTensor->getDnnLayout();
        dnnLayout_t resultLayout;
        dnnError_t err;

        if (reluPrim == NULL)
        {
            err = dnn::xReLUCreateBackward( &reluPrim, inputGradLayout, inputLayout, (algorithmFPType)0.0); ON_ERR(err);
        }

        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, reluPrim, dnnResourceDiffSrc); ON_ERR(err);
        resultMklTensor->setDnnLayout(resultLayout);

        algorithmFPType* reluRes[dnnResourceNumber] = {0};

        reluRes[dnnResourceDiffDst] = inputGradientMklTensor->getDnnArray();
        reluRes[dnnResourceDiffSrc] = resultMklTensor->getDnnArray();
        reluRes[dnnResourceSrc] = forwardDataMklTensor->getDnnArray();

        err = dnn::xExecute(reluPrim, (void**)reluRes); ON_ERR(err);
    }
    else
    {
        __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&forwardDataTensor))
        __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)

        s = computeImpl<cpu>(inputGradientTensor, [=, &inputGradientTensor, &forwardDataTensor, &resultTensor](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout) -> Status
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(const_cast<Tensor &>(inputGradientTensor), fDimN, fDims, 0, nRowsToProcess, layout);
            DAAL_CHECK_BLOCK_STATUS(inputGradientBlock);
            const algorithmFPType *inputGradientArray = inputGradientBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> forwardBlock(const_cast<Tensor &>(forwardDataTensor), fDimN, fDims, 0, nRowsToProcess, layout);
            DAAL_CHECK_BLOCK_STATUS(forwardBlock);
            const algorithmFPType *forwardDataArray = forwardBlock.get();

            WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);
            algorithmFPType *resultArray = resultBlock.get();

            algorithmFPType zero = (algorithmFPType)0;
            size_t nDataElements = inputGradientBlock.getSize();

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < nDataElements; i++)
            {
                if(forwardDataArray[i] > zero)
                {
                    resultArray[i] = inputGradientArray[i];
                }
                else
                {
                    resultArray[i] = zero;
                }
            }
            return Status();
        });
    }
    return s;
}

} // namespace internal
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
