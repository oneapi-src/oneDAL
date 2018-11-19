/* file: logistic_layer_forward_impl.i */
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
//  Implementation of logistic function algorithm
//--
*/

#ifndef __LOGISTIC_LAYER_FORWARD_IMPL_I__
#define __LOGISTIC_LAYER_FORWARD_IMPL_I__

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
namespace logistic
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LogisticKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, Tensor &resultTensor)
{
    __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)

    Status s = computeImpl<cpu>(inputTensor, [=, &inputTensor, &resultTensor](size_t fDimN, size_t *fDims, size_t nRowsToProcess, const TensorOffsetLayout &layout) -> Status
    {
        ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), fDimN, fDims, 0, nRowsToProcess, layout);
        DAAL_CHECK_BLOCK_STATUS(inputBlock);
        const algorithmFPType *inputArray = inputBlock.get();

        WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, fDimN, fDims, 0, nRowsToProcess, layout);
        DAAL_CHECK_BLOCK_STATUS(resultBlock);
        algorithmFPType *resultArray = resultBlock.get();

        size_t nDataElements = inputBlock.getSize();
        algorithmFPType one = (algorithmFPType)1.0;

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = - inputArray[i];

            /* Arguments filtering before vector exponential function call */
            /* There is a known issue that vExp works slowly on large negative arguments (where results are zero or denormals) */
            if( resultArray[i] < daal::internal::Math<algorithmFPType,cpu>::vExpThreshold() )
            {
                resultArray[i] = daal::internal::Math<algorithmFPType,cpu>::vExpThreshold();
            }
        }

        daal::internal::Math<algorithmFPType,cpu>::vExp(nDataElements, resultArray, resultArray);

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = one / ( one + resultArray[i] );
        }
        return Status();
    });
    return s;

}

} // internal
} // forward
} // namespace logistic
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
