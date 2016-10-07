/* file: relu_dense_default_impl.i */
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
//  Implementation of relu algorithm
//--
*/

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace relu
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
inline void ReLUKernel<algorithmFPType, defaultDense, cpu>::processBlock(NumericTable *inputTable, size_t nInputColumns,
                                                                         size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                         NumericTable *resultTable)
{
    BlockDescriptor<algorithmFPType> inputBlock;
    inputTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> resultBlock;
    resultTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getBlockPtr();

    size_t nDataElements = nRowsInCurrentBlock * nInputColumns;
    for(size_t i = 0; i < nDataElements; i++)
    {
        if(inputArray[i] >= (algorithmFPType)0)
        {
            resultArray[i] = inputArray[i];
        }
        else
        {
            resultArray[i] = (algorithmFPType)0;
        }
    }

    inputTable->releaseBlockOfRows(inputBlock);
    resultTable->releaseBlockOfRows(resultBlock);
}

} // namespace daal::internal
} // namespace relu
} // namespace math
} // namespace algorithms
} // namespace daal
