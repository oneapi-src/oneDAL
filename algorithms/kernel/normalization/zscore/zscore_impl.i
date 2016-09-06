/* file: zscore_impl.i */
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

//++
//  Implementation of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_IMPL_I__
#define __ZSCORE_IMPL_I__

#include "zscore_base.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
void ZScoreKernelBase<algorithmFPType, cpu>::compute(const Input *input, Result *result, daal::algorithms::Parameter *parameter)
{
    algorithmFPType *meanArray;
    algorithmFPType *standardDeviationInverse;

    NumericTablePtr inputTable  = input->get(data);
    NumericTablePtr resultTable = result->get(normalizedData);

    size_t nInputRows    = inputTable->getNumberOfRows();
    size_t nInputColumns = inputTable->getNumberOfColumns();

    size_t nBlocks = nInputRows / _nRowsInBlock;
    size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

    if (inputTable->isNormalized(NumericTableIface::standardScoreNormalized))
    {
        checkForInplace(inputTable, resultTable, nInputColumns, nBlocks, nRowsInLastBlock);
        return;
    }

    standardDeviationInverse = (algorithmFPType *) daal_malloc(nInputColumns * sizeof(algorithmFPType));
    if (!standardDeviationInverse) { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    computeInternal(inputTable, nInputRows, nInputColumns, nBlocks, nRowsInLastBlock, parameter, &meanArray, standardDeviationInverse);

    for(size_t block = 0; block < nBlocks; block++)
    {
        normalizeDataInBlock(inputTable, nInputColumns, block * _nRowsInBlock, _nRowsInBlock, resultTable, meanArray,
                             standardDeviationInverse);
    }
    if(nRowsInLastBlock > 0)
    {
        normalizeDataInBlock(inputTable, nInputColumns, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTable, meanArray,
                             standardDeviationInverse);
    }

    resultTable->setNormalizationFlag(NumericTableIface::standardScoreNormalized);

    daal_free(standardDeviationInverse);
    releaseData(meanArray);
};

template<typename algorithmFPType, CpuType cpu>
inline void ZScoreKernelBase<algorithmFPType, cpu>::checkForInplace(NumericTablePtr inputTable, NumericTablePtr resultTable,
                                                                    size_t nInputColumns, size_t nBlocks, size_t nRowsInLastBlock)
{
    if(inputTable.get() == resultTable.get())
    {
        return;
    }
    else
    {
        for(size_t block = 0; block < nBlocks; block++)
        {
            copyDataBlock(inputTable, nInputColumns, block * _nRowsInBlock, _nRowsInBlock, resultTable);
        }
        if(nRowsInLastBlock > 0)
        {
            copyDataBlock(inputTable, nInputColumns, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTable);
        }

        resultTable->setNormalizationFlag(NumericTableIface::standardScoreNormalized);
    }
}

template<typename algorithmFPType, CpuType cpu>
inline void ZScoreKernelBase<algorithmFPType, cpu>::copyDataBlock(NumericTablePtr inputTable, size_t nInputColumns, size_t nProcessedRows,
                                                                  size_t nRowsInCurrentBlock, NumericTablePtr resultTable)
{
    BlockDescriptor<algorithmFPType> inputBlock;
    inputTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> resultBlock;
    resultTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getBlockPtr();

    for(size_t i = 0; i < nRowsInCurrentBlock * nInputColumns; i++)
    {
        resultArray[i] = inputArray[i];
    }

    inputTable->releaseBlockOfRows(inputBlock);
    resultTable->releaseBlockOfRows(resultBlock);
}

template<typename algorithmFPType, CpuType cpu>
inline void ZScoreKernelBase<algorithmFPType, cpu>::normalizeDataInBlock(NumericTablePtr inputTable, size_t nInputColumns,
                                                                         size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                         NumericTablePtr resultTable, algorithmFPType *meanArray,
                                                                         algorithmFPType *standardDeviationInverse)
{
    BlockDescriptor<algorithmFPType> inputBlock;
    inputTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> resultBlock;
    resultTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getBlockPtr();

    for(size_t j = 0; j < nRowsInCurrentBlock; j++)
    {
        for(size_t i = 0; i < nInputColumns; i++)
        {
            resultArray[j * nInputColumns + i] = ( inputArray[j * nInputColumns + i] - meanArray[i] ) * standardDeviationInverse[i];
        }
    }

    inputTable->releaseBlockOfRows(inputBlock);
    resultTable->releaseBlockOfRows(resultBlock);
}

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
