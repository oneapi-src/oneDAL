/* file: logistic_cross_layer_forward_impl.i */
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
//  Implementation of the forward logistic cross layer
//--
*/

#ifndef __LOGISTIC_CROSS_LAYER_FORWARD_IMPL_I__
#define __LOGISTIC_CROSS_LAYER_FORWARD_IMPL_I__

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
namespace loss
{
namespace logistic_cross
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LogisticCrossKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const Tensor &groundTruthTensor, Tensor &resultTensor)
{
    size_t nRowsToProcess = inputTensor.getDimensionSize(0);
    TArray<algorithmFPType, cpu> sPtr(nRowsToProcess);
    algorithmFPType *s = sPtr.get();

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor&>(inputTensor), 0, 0, 0, nRowsToProcess);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *inputArray = inputBlock.get();

    ReadSubtensor<algorithmFPType, cpu, Tensor> groundTruthBlock(const_cast<Tensor&>(groundTruthTensor), 0, 0, 0, nRowsToProcess);
    DAAL_CHECK_BLOCK_STATUS(groundTruthBlock);
    const algorithmFPType *groundTruthArray = groundTruthBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, 0, 0, 0, nRowsToProcess);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType &loss = resultBlock.get()[0];

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        if(inputArray[i] >= (algorithmFPType)0)
        {
            s[i] = -inputArray[i];
        }
        else
        {
            s[i] = inputArray[i];
        }
    }
    Math<algorithmFPType, cpu>::vExp(nDataElements, s, s);
    for(size_t i = 0; i < nDataElements; i++)
    {
        s[i] += 1.0;
    }
    Math<algorithmFPType, cpu>::vLog(nDataElements, s, s);
    for(size_t i = 0; i < nDataElements; i++)
    {
        s[i] = inputArray[i] * ((inputArray[i] > 0) - groundTruthArray[i]) + s[i];
    }

    loss = 0;
    for(size_t i = 0; i < nDataElements; i++)
    {
        loss += s[i];
    }
    loss = loss / nRowsToProcess;
    return services::Status();
}

} // namespace internal
} // namespace forward
} // namespace logistic_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
