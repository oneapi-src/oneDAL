/* file: smoothrelu_impl.i */
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
//  Implementation of smoothrelu algorithm
//--
*/

namespace daal
{
namespace algorithms
{
namespace math
{
namespace smoothrelu
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
inline Status SmoothReLUKernel<algorithmFPType, method, cpu>::processBlock(const NumericTable &inputTable,
                                                                           size_t nInputColumns,
                                                                           size_t nProcessedRows,
                                                                           size_t nRowsInCurrentBlock,
                                                                           NumericTable &resultTable)
{
    ReadRows<algorithmFPType, cpu, NumericTable> inputBlock(const_cast<NumericTable *>(&inputTable), nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType* inputArray = inputBlock.get();

    WriteRows<algorithmFPType, cpu, NumericTable> resultBlock(&resultTable, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType* resultArray = resultBlock.get();

    const algorithmFPType one = (algorithmFPType)1.0;
    const size_t nDataElements = nRowsInCurrentBlock * nInputColumns;

    /* res = log(1+exp(in)) */
    daal::internal::Math<algorithmFPType, cpu>::vExp(nDataElements, const_cast<algorithmFPType *>(inputArray), resultArray);
    daal::internal::Math<algorithmFPType, cpu>::vLog1p(nDataElements, resultArray, resultArray);
    return Status();
}

/**
 *  \brief Kernel for SmoothReLU calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status SmoothReLUKernel<algorithmFPType, method, cpu>::compute(const NumericTable *inputTable, NumericTable *resultTable)
{
    const size_t nInputRows = inputTable->getNumberOfRows();
    const size_t nInputColumns = inputTable->getNumberOfColumns();

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [ =, &safeStat ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }

        safeStat |= processBlock(*inputTable, nInputColumns, block * _nRowsInBlock, nRowsToProcess, *resultTable);
    } );
    return safeStat.detach();
}

} // namespace daal::internal
} // namespace smoothrelu
} // namespace math
} // namespace algorithms
} // namespace daal
