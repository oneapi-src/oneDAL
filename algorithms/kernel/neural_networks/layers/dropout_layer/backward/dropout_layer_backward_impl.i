/* file: dropout_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the backward dropout layer
//--
*/

#ifndef __DROPOUT_LAYER_BACKWARD_IMPL_I__
#define __DROPOUT_LAYER_BACKWARD_IMPL_I__

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
namespace dropout
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status DropoutKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputGradientTable,
        const Tensor &maskTable,
        Tensor &resultTable)
{
    const size_t nInputRows = inputGradientTable.getDimensionSize(0);
    const size_t nBlocks = nInputRows / _nRowsInBlock;
    const size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    Status s;
    for(size_t block = 0; block < nBlocks; block++)
    {
        s |= processBlock(inputGradientTable, maskTable, block * _nRowsInBlock, _nRowsInBlock, resultTable);
    }
    if(nRowsInLastBlock > 0)
    {
        s |= processBlock(inputGradientTable, maskTable, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTable);
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline Status DropoutKernel<algorithmFPType, method, cpu>::processBlock(
    const Tensor &inputGradientTable,
    const Tensor &maskTable,
    const size_t nProcessedRows,
    const size_t nRowsInCurrentBlock,
    Tensor &resultTable)
{
    ReadSubtensor<algorithmFPType, cpu> inputGradientBlock(const_cast<Tensor&>(inputGradientTable), 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputGradientBlock);
    const algorithmFPType *inputGradientArray = inputGradientBlock.get();

    ReadSubtensor<algorithmFPType, cpu> maskBlock(const_cast<Tensor&>(maskTable), 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(maskBlock);
    const algorithmFPType *maskArray = maskBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> resultBlock(resultTable, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    const size_t nDataElements = inputGradientBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputGradientArray[i] * maskArray[i];
    }

    return Status();
}

} // internal
} // backward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
