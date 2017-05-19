/* file: relu_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
services::Status ReLUKernel<algorithmFPType, method, cpu>::compute(Tensor *inputGradientTensor, Tensor *forwardDataTensor, Tensor *resultTensor)
{
    MklTensor<algorithmFPType> *inputGradientMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputGradientTensor);
    MklTensor<algorithmFPType> *forwardDataMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(forwardDataTensor);
    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(resultTensor);
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
        __DAAL_MAKE_TENSOR_THREADSAFE(forwardDataTensor)
        __DAAL_MAKE_TENSOR_THREADSAFE(resultTensor)

        computeImpl<cpu>(inputGradientTensor, this->_errors.get(), [=](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout)
        {
            ReadSubtensor<algorithmFPType, cpu, Tensor> inputGradientBlock(*inputGradientTensor, fDimN, fDims, 0, nRowsToProcess, layout);
            const algorithmFPType *inputGradientArray = inputGradientBlock.get();

            ReadSubtensor<algorithmFPType, cpu, Tensor> forwardBlock(*forwardDataTensor, fDimN, fDims, 0, nRowsToProcess, layout);
            const algorithmFPType *forwardDataArray = forwardBlock.get();

            WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(*resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
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
        });
    }
    DAAL_RETURN_STATUS();
}

} // namespace internal
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
