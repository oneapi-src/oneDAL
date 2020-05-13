/** file train_test_split.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "data_management/data/internal/train_test_split.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "service_dispatch.h"
#include "service_data_utils.h"
#include "threading.h"
#include "service_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const size_t BLOCK_CONST = 10240;
const size_t THREADING_BORDER = 12800000;

template <typename Func>
void runBlocks(const bool inParallel, const size_t nBlocks, Func func)
{
    if (inParallel)
        daal::threader_for(nBlocks, nBlocks, [&](size_t i) { func(i); });
    else
        for (size_t i = 0; i < nBlocks; ++i) func(i);
}

template <typename DataType, typename IdxType>
services::Status splitColumn(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                             const IdxType * trainIdx, const IdxType * testIdx,
                             const size_t nTrainRows, const size_t nTestRows, const size_t iCol)
{
    services::Status s;
    BlockDescriptor<DataType> origBlock, trainBlock, testBlock;
    inputTable.getBlockOfColumnValues(iCol, 0, nTrainRows + nTestRows, readOnly, origBlock);
    trainTable.getBlockOfColumnValues(iCol, 0, nTrainRows, readWrite, trainBlock);
    testTable.getBlockOfColumnValues(iCol, 0, nTestRows, readWrite, testBlock);
    const DataType * origDataPtr = origBlock.getBlockPtr();
    DataType * trainDataPtr = trainBlock.getBlockPtr();
    DataType * testDataPtr = testBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(origDataPtr);
    DAAL_CHECK_MALLOC(trainDataPtr);
    DAAL_CHECK_MALLOC(testDataPtr);

    const size_t BLOCK_SIZE = BLOCK_CONST;
    const size_t nTrainBlocks = nTrainRows / BLOCK_SIZE ? nTrainRows / BLOCK_SIZE : 1;
    const size_t nTestBlocks = nTestRows / BLOCK_SIZE ? nTestRows / BLOCK_SIZE : 1;
    const int nTrainSurplus = nTrainRows / BLOCK_SIZE ? nTrainRows % BLOCK_SIZE : nTrainRows - BLOCK_SIZE;
    const int nTestSurplus = nTestRows / BLOCK_SIZE ? nTestRows % BLOCK_SIZE : nTestRows - BLOCK_SIZE;

    runBlocks(nTrainRows > THREADING_BORDER, nTrainBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * BLOCK_SIZE;
        const size_t end   = iBlock == nTrainBlocks - 1 ? start + BLOCK_SIZE + nTrainSurplus : start + BLOCK_SIZE;

        for (size_t i = start; i < end; ++i)
            trainDataPtr[i] = origDataPtr[trainIdx[i]];
    });

    runBlocks(nTestRows > THREADING_BORDER, nTestBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * BLOCK_SIZE;
        const size_t end   = iBlock == nTestBlocks - 1 ? start + BLOCK_SIZE + nTestSurplus : start + BLOCK_SIZE;

        for (size_t i = start; i < end; ++i)
            testDataPtr[i] = origDataPtr[testIdx[i]];
    });

    return s;
}

template <typename DataType, typename IdxType>
services::Status splitRows(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                           const IdxType * trainIdx, const IdxType * testIdx,
                           const size_t nTrainRows, const size_t nTestRows, const size_t nColumns)
{
    services::Status s;
    const size_t BLOCK_SIZE = BLOCK_CONST / nColumns ? BLOCK_CONST / nColumns : 1;
    const size_t nTrainBlocks = nTrainRows / BLOCK_SIZE ? nTrainRows / BLOCK_SIZE : 1;
    const size_t nTestBlocks = nTestRows / BLOCK_SIZE ? nTestRows / BLOCK_SIZE : 1;
    const int nTrainSurplus = nTrainRows / BLOCK_SIZE ? nTrainRows % BLOCK_SIZE : nTrainRows - BLOCK_SIZE;
    const int nTestSurplus = nTestRows / BLOCK_SIZE ? nTestRows % BLOCK_SIZE : nTestRows - BLOCK_SIZE;

    BlockDescriptor<DataType> origBlock, trainBlock, testBlock;
    inputTable.getBlockOfRows(0, nTrainRows + nTestRows, readOnly, origBlock);
    trainTable.getBlockOfRows(0, nTrainRows, readWrite, trainBlock);
    testTable.getBlockOfRows(0, nTestRows, readWrite, testBlock);
    const DataType * origDataPtr = origBlock.getBlockPtr();
    DataType * trainDataPtr = trainBlock.getBlockPtr();
    DataType * testDataPtr = testBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(origDataPtr);
    DAAL_CHECK_MALLOC(trainDataPtr);
    DAAL_CHECK_MALLOC(testDataPtr);

    runBlocks(nTrainRows * nColumns > THREADING_BORDER, nTrainBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * BLOCK_SIZE;
        const size_t end   = iBlock == nTrainBlocks - 1 ? start + BLOCK_SIZE + nTrainSurplus : start + BLOCK_SIZE;

        for (size_t i = start; i < end; ++i)
            for (size_t j = 0; j < nColumns; ++j)
                trainDataPtr[i * nColumns + j] = origDataPtr[trainIdx[i] * nColumns + j];
    });

    runBlocks(nTestRows * nColumns > THREADING_BORDER, nTestBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * BLOCK_SIZE;
        const size_t end   = iBlock == nTestBlocks - 1 ? start + BLOCK_SIZE + nTestSurplus : start + BLOCK_SIZE;

        for (size_t i = start; i < end; ++i)
            for (size_t j = 0; j < nColumns; ++j)
                testDataPtr[i * nColumns + j] = origDataPtr[testIdx[i] * nColumns + j];
    });
    return s;
}


template <typename IdxType>
services::Status trainTestSplitImpl(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                                    NumericTable & trainIdxTable, NumericTable & testIdxTable, NumericTable & columnTypesTable)
{
    services::Status s;
    const NTLayout layout  = inputTable.getDataLayout();
    const size_t nColumns = trainTable.getNumberOfColumns();
    const size_t nTrainRows = trainTable.getNumberOfRows();
    const size_t nTestRows = testTable.getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrainRows + nTestRows, nColumns);

    BlockDescriptor<int> columnTypesBlock;
    columnTypesTable.getBlockOfRows(0, nColumns, readOnly, columnTypesBlock);
    const int * columnTypesPtr = columnTypesBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(columnTypesPtr);

    BlockDescriptor<IdxType> trainIdxBlock, testIdxBlock;
    trainIdxTable.getBlockOfColumnValues(0, 0, nTrainRows, readOnly, trainIdxBlock);
    testIdxTable.getBlockOfColumnValues(0, 0, nTestRows, readOnly, testIdxBlock);
    const IdxType * trainIdx = trainIdxBlock.getBlockPtr();
    const IdxType * testIdx = testIdxBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(trainIdx);
    DAAL_CHECK_MALLOC(testIdx);

    if (layout == NTLayout::soa)
    {
        runBlocks(nColumns > 1 && nColumns * (nTrainRows + nTestRows) > THREADING_BORDER, nColumns, [&](size_t iCol) {
            switch (columnTypesPtr[iCol])
            {
                case 0:
                    splitColumn<int, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol);
                    break;
                case 1:
                    splitColumn<float, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol);
                    break;
                case 2:
                    splitColumn<double, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol);
                    break;
            }
        });
    }
    else
    {
        switch (columnTypesPtr[0])
        {
            case 0:
                splitRows<int, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, nColumns);
                break;
            case 1:
                splitRows<float, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, nColumns);
                break;
            case 2:
                splitRows<double, IdxType>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, nColumns);
                break;
        }
    }
    return s;
}

template <typename IdxType>
DAAL_EXPORT void trainTestSplit(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                                NumericTable & trainIdxTable, NumericTable & testIdxTable, NumericTable & columnTypesTable)
{
    services::Status s = trainTestSplitImpl<IdxType>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, columnTypesTable);
}

template DAAL_EXPORT void trainTestSplit<int>(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                                              NumericTable & trainIdxTable, NumericTable & testIdxTable, NumericTable & columnTypesTable);

} // namespace internal
} // namespace data_management
} // namespace daal
