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
#include "data_management/data/data_dictionary.h"
#include "services/env_detect.h"
#include "externals/service_dispatch.h"
#include "service/kernel/service_data_utils.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_defines.h"
#include "algorithms/threading/threading.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "data_management/features/defines.h"

namespace daal
{
namespace data_management
{
namespace internal
{
typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const size_t BLOCK_CONST      = 2048;
const size_t THREADING_BORDER = 8388608;

template <typename Func>
void runBlocks(const bool inParallel, const size_t nBlocks, Func func)
{
    if (inParallel)
        daal::threader_for(nBlocks, nBlocks, [&](size_t i) { func(i); });
    else
        for (size_t i = 0; i < nBlocks; ++i) func(i);
}

template <typename DataType, typename IdxType>
services::Status assignColumnValues(const DataType * origDataPtr, NumericTable & dataTable, const IdxType * idxPtr, const size_t startRow,
                                    const size_t nRows, const size_t iCol)
{
    services::Status s;

    BlockDescriptor<DataType> dataBlock;
    dataTable.getBlockOfColumnValues(iCol, startRow, nRows, readWrite, dataBlock);
    DataType * dataPtr = dataBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(dataPtr);

    for (size_t i = 0; i < nRows; ++i) dataPtr[i] = origDataPtr[idxPtr[i]];

    return s;
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitColumn(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable, const IdxType * trainIdx,
                             const IdxType * testIdx, const size_t nTrainRows, const size_t nTestRows, const size_t iCol, const size_t nThreads)
{
    services::Status s;
    const size_t BLOCK_SIZE = BLOCK_CONST;

    BlockDescriptor<DataType> origBlock, trainBlock, testBlock;
    inputTable.getBlockOfColumnValues(iCol, 0, nTrainRows + nTestRows, readOnly, origBlock);
    const DataType * origDataPtr = origBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(origDataPtr);

    if (nTrainRows > THREADING_BORDER && nThreads > 1)
    {
        const size_t nTrainBlocks = daal::services::internal::max<cpu, size_t>(nTrainRows / BLOCK_SIZE, 1);
        const int nTrainSurplus   = nTrainRows - BLOCK_SIZE * nTrainBlocks;

        daal::threader_for(nTrainBlocks, nTrainBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_SIZE;
            const size_t end   = iBlock == nTrainBlocks - 1 ? start + BLOCK_SIZE + nTrainSurplus : start + BLOCK_SIZE;

            assignColumnValues<DataType, IdxType>(origDataPtr, trainTable, trainIdx, start, end - start, iCol);
        });
    }
    else
    {
        assignColumnValues<DataType, IdxType>(origDataPtr, trainTable, trainIdx, 0, nTrainRows, iCol);
    }

    if (nTestRows > THREADING_BORDER && nThreads > 1)
    {
        const size_t nTestBlocks = daal::services::internal::max<cpu, size_t>(nTestRows / BLOCK_SIZE, 1);
        const int nTestSurplus   = nTestRows - BLOCK_SIZE * nTestBlocks;

        daal::threader_for(nTestBlocks, nTestBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_SIZE;
            const size_t end   = iBlock == nTestBlocks - 1 ? start + BLOCK_SIZE + nTestSurplus : start + BLOCK_SIZE;

            assignColumnValues<DataType, IdxType>(origDataPtr, testTable, testIdx, start, end - start, iCol);
        });
    }
    else
    {
        assignColumnValues<DataType, IdxType>(origDataPtr, testTable, testIdx, 0, nTestRows, iCol);
    }

    return s;
}

template <typename DataType, typename IdxType>
services::Status assignRows(const DataType * origDataPtr, NumericTable & dataTable, NumericTable & idxTable, const size_t startRow,
                            const size_t nRows, const size_t nColumns)
{
    services::Status s;
    BlockDescriptor<DataType> dataBlock;
    BlockDescriptor<IdxType> idxBlock;
    dataTable.getBlockOfRows(startRow, startRow + nRows, readWrite, dataBlock);
    idxTable.getBlockOfColumnValues(0, startRow, nRows, readOnly, idxBlock);
    DataType * dataPtr     = dataBlock.getBlockPtr();
    const IdxType * idxPtr = idxBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(dataPtr);
    DAAL_CHECK_MALLOC(idxPtr);

    for (size_t i = 0; i < nRows; ++i)
    {
        const IdxType idx = idxPtr[i];
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nColumns; ++j) dataPtr[i * nColumns + j] = origDataPtr[idxPtr[i] * nColumns + j];
    }
    return s;
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitRows(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable, NumericTable & trainIdxTable,
                           NumericTable & testIdxTable, const size_t nTrainRows, const size_t nTestRows, const size_t nColumns, const size_t nThreads)
{
    services::Status s;
    const size_t BLOCK_SIZE = daal::services::internal::max<cpu, size_t>(BLOCK_CONST / nColumns, 1);
    BlockDescriptor<DataType> origBlock;
    inputTable.getBlockOfRows(0, nTrainRows + nTestRows, readOnly, origBlock);
    const DataType * origDataPtr = origBlock.getBlockPtr();
    DAAL_CHECK_MALLOC(origDataPtr);

    if (nTrainRows * nColumns > THREADING_BORDER && nThreads > 1)
    {
        const size_t nTrainBlocks = daal::services::internal::max<cpu, size_t>(nTrainRows / BLOCK_SIZE, 1);
        const int nTrainSurplus   = nTrainRows - BLOCK_SIZE * nTrainBlocks;

        daal::threader_for(nTrainBlocks, nTrainBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_SIZE;
            const size_t end   = iBlock == nTrainBlocks - 1 ? start + BLOCK_SIZE + nTrainSurplus : start + BLOCK_SIZE;

            s |= assignRows<DataType, IdxType>(origDataPtr, trainTable, trainIdxTable, start, end - start, nColumns);
        });
    }
    else
    {
        s |= assignRows<DataType, IdxType>(origDataPtr, trainTable, trainIdxTable, 0, nTrainRows, nColumns);
    }

    if (nTestRows * nColumns > THREADING_BORDER && nThreads > 1)
    {
        const size_t nTestBlocks = daal::services::internal::max<cpu, size_t>(nTestRows / BLOCK_SIZE, 1);
        const int nTestSurplus   = nTestRows - BLOCK_SIZE * nTestBlocks;

        daal::threader_for(nTestBlocks, nTestBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_SIZE;
            const size_t end   = iBlock == nTestBlocks - 1 ? start + BLOCK_SIZE + nTestSurplus : start + BLOCK_SIZE;

            s |= assignRows<DataType, IdxType>(origDataPtr, testTable, testIdxTable, start, end - start, nColumns);
        });
    }
    else
    {
        s |= assignRows<DataType, IdxType>(origDataPtr, testTable, testIdxTable, 0, nTestRows, nColumns);
    }

    return s;
}

template <typename IdxType, daal::CpuType cpu>
services::Status trainTestSplitImpl(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable, NumericTable & trainIdxTable,
                                    NumericTable & testIdxTable)
{
    services::Status s;
    const size_t nThreads   = threader_get_threads_number();
    const NTLayout layout   = inputTable.getDataLayout();
    const size_t nColumns   = trainTable.getNumberOfColumns();
    const size_t nTrainRows = trainTable.getNumberOfRows();
    const size_t nTestRows  = testTable.getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrainRows + nTestRows, nColumns);

    NumericTableDictionaryPtr tableFeaturesDict = inputTable.getDictionarySharedPtr();

    if (layout == NTLayout::soa)
    {
        BlockDescriptor<IdxType> trainIdxBlock, testIdxBlock;
        trainIdxTable.getBlockOfColumnValues(0, 0, nTrainRows, readOnly, trainIdxBlock);
        testIdxTable.getBlockOfColumnValues(0, 0, nTestRows, readOnly, testIdxBlock);
        const IdxType * trainIdx = trainIdxBlock.getBlockPtr();
        const IdxType * testIdx  = testIdxBlock.getBlockPtr();
        DAAL_CHECK_MALLOC(trainIdx);
        DAAL_CHECK_MALLOC(testIdx);

        runBlocks(nColumns > 1 && nColumns * (nTrainRows + nTestRows) > THREADING_BORDER && nThreads > 1, nColumns, [&](size_t iCol) {
            switch ((*tableFeaturesDict)[iCol].getIndexType())
            {
            case daal::data_management::features::IndexNumType::DAAL_INT32_S:
                s |= splitColumn<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                break;
            case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
                s |= splitColumn<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                break;
            case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
                s |= splitColumn<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                break;
            }
        });
    }
    else
    {
        switch ((*tableFeaturesDict)[0].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_INT32_S:
            s |= splitRows<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                              nThreads);
            break;
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
            s |= splitRows<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                nThreads);
            break;
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
            s |= splitRows<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                 nThreads);
            break;
        }
    }
    return s;
}

template <typename IdxType>
void trainTestSplitDispImpl(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable, NumericTable & trainIdxTable,
                            NumericTable & testIdxTable)
{
#define DAAL_TTS(cpuId, ...) trainTestSplitImpl<IdxType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_TTS, inputTable, trainTable, testTable, trainIdxTable, testIdxTable);
#undef DAAL_TTS
}

template <typename IdxType>
DAAL_EXPORT void trainTestSplit(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable, NumericTable & trainIdxTable,
                                NumericTable & testIdxTable)
{
    DAAL_SAFE_CPU_CALL((trainTestSplitDispImpl<IdxType>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)),
                       (trainTestSplitImpl<IdxType, daal::CpuType::sse2>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)));
}

template DAAL_EXPORT void trainTestSplit<int>(NumericTable & inputTable, NumericTable & trainTable, NumericTable & testTable,
                                              NumericTable & trainIdxTable, NumericTable & testIdxTable);

} // namespace internal
} // namespace data_management
} // namespace daal
