/* file: softmax_impl.i */
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
//  Implementation of softmax algorithm
//--
*/

namespace daal
{
namespace algorithms
{
namespace math
{
namespace softmax
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
inline Status SoftmaxKernel<algorithmFPType, method, cpu>::processBlock(const NumericTable &inputTable, size_t nInputColumns,
                                                                        size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                        NumericTable &resultTable)
{
    ReadRows<algorithmFPType, cpu, NumericTable> inputBlock(const_cast<NumericTable *>(&inputTable), nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType* inputArray = inputBlock.get();

    WriteRows<algorithmFPType, cpu, NumericTable> resultBlock(&resultTable, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType* resultArray = resultBlock.get();

    algorithmFPType minValue = -services::internal::MaxVal<algorithmFPType>::get();
    algorithmFPType max;
    algorithmFPType sum = (algorithmFPType)0;

    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        max = minValue;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nInputColumns; j++)
        {
            if(max < inputArray[i * nInputColumns + j])
            {
                max = inputArray[i * nInputColumns + j];
            }
            resultArray[i * nInputColumns + j] = inputArray[i * nInputColumns + j];
        }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] -= max;

            /* make all values less than threshold as threshold value
               to fix slow work on vExp on large negative inputs */
            if( resultArray[i * nInputColumns + j] < daal::internal::Math<algorithmFPType, cpu>::vExpThreshold() )
            {
                resultArray[i * nInputColumns + j] = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
            }
        }

        daal::internal::Math<algorithmFPType, cpu>::vExp(nInputColumns, resultArray + i * nInputColumns, resultArray + i * nInputColumns);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nInputColumns; j++)
        {
            sum += resultArray[i * nInputColumns + j];
        }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] /= sum;
        }

        sum = (algorithmFPType)0;
    }
    return Status();
}

/**
 *  \brief Kernel for Softmax calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SoftmaxKernel<algorithmFPType, method, cpu>::compute(const NumericTable *inputTable, NumericTable *resultTable)
{
    const size_t nInputRows    = inputTable->getNumberOfRows();
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
} // namespace softmax
} // namespace math
} // namespace algorithms
} // namespace daal
