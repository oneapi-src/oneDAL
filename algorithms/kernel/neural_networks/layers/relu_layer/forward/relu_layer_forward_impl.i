/* file: relu_layer_forward_impl.i */
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
//  Implementation of relu algorithm
//--
*/

#include "service_tensor.h"
#include "service_dnn.h"
#include "service_mkl_tensor.h"

#ifndef __RELU_LAYER_FORWARD_IMPL_I__
#define __RELU_LAYER_FORWARD_IMPL_I__

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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status ReLUKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, Tensor &resultTensor)
{
    MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(const_cast<Tensor *>(&inputTensor));
    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(&resultTensor);

    Status s;

    if (inputMklTensor != 0 && resultMklTensor != 0)
    {
        dnnLayout_t inputLayout = (dnnLayout_t)inputMklTensor->getDnnLayout();
        dnnLayout_t resultLayout;

        dnnError_t err;

        if (reluPrim == NULL)
        {
            err = dnn::xReLUCreateForward( &reluPrim, inputLayout, (algorithmFPType)0.0); ON_ERR(err);
        }

        if (inputMklTensor != resultMklTensor)
        {
            err = dnn::xLayoutCreateFromPrimitive(&resultLayout, reluPrim, dnnResourceDst); ON_ERR(err);
            resultMklTensor->setDnnLayout(resultLayout);
        }

        algorithmFPType* reluRes[dnnResourceNumber] = {0};

        reluRes[dnnResourceSrc] = inputMklTensor->getDnnArray();
        reluRes[dnnResourceDst] = resultMklTensor->getDnnArray();

        err = dnn::xExecute(reluPrim, (void**)reluRes); ON_ERR(err);
    }
    else
    {
        __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)

        s = computeImpl<cpu>(inputTensor, [=, &inputTensor, &resultTensor](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout) -> Status
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), fDimN, fDims, 0, nRowsToProcess, layout);
            DAAL_CHECK_BLOCK_STATUS(inputBlock);
            const algorithmFPType *inputArray = inputBlock.get();

            WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);
            algorithmFPType *resultArray = resultBlock.get();

            algorithmFPType zero = (algorithmFPType)0;
            size_t nDataElements = inputBlock.getSize();
            for(size_t i = 0; i < nDataElements; i++)
            {
                if(inputArray[i] > zero)
                {
                    resultArray[i] = inputArray[i];
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
} // namespace forward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
