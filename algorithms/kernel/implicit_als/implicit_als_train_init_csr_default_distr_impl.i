/* file: implicit_als_train_init_csr_default_distr_impl.i */
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
//  Implementation of impicit ALS model initialization in distributed mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__

#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_spblas.h"
#include "service_sort.h"
#include "implicit_als_train_dense_default_batch_common.i"
#include "implicit_als_train_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{

using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::internal;

template <CpuType cpu>
class Partition
{
public:
    Partition(): nParts(0), _partition(nullptr){}
    Status init(NumericTable *partitionTable, size_t fullNUsers)
    {
        const size_t nRows = partitionTable->getNumberOfRows();
        _partitionRows.set(partitionTable, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(_partitionRows);
        if (nRows > 1)
        {
            nParts = nRows - 1;
            _partition = const_cast<int *>(_partitionRows.get());
        }
        else
        {
            nParts = (size_t)((_partitionRows.get())[0]);
            _partitionPtr.reset(nParts + 1);
            DAAL_CHECK_MALLOC(_partitionPtr.get());
            _partition = _partitionPtr.get();
            const size_t nUsersInPart = fullNUsers / nParts;
            _partition[0] = 0;
            for (size_t i = 1; i < nParts; i++)
            {
                _partition[i] = _partition[i - 1] + nUsersInPart;
            }
            _partition[nParts] = fullNUsers;
        }
        return Status();
    }

    int *get()
    {
        return _partition;
    }

    size_t nParts;

private:
    ReadRows<int, cpu> _partitionRows;
    TArray<int, cpu> _partitionPtr;
    int *_partition;
};

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::compute(
            const NumericTable *dataTable, const NumericTable *partitionTable,
            NumericTable **dataParts, NumericTable **blocksToLocal,
            NumericTable **userOffsets, NumericTable *itemsFactorsTable, const DistributedParameter *parameter, engines::BatchBase &engine)
{
    const size_t nItems = dataTable->getNumberOfRows();
    const size_t nUsers = dataTable->getNumberOfColumns();
    const size_t fullNUsers = parameter->fullNUsers;
    ReadRowsCSR<algorithmFPType, cpu> mtData(
        dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(dataTable)), 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtData);
    const algorithmFPType *tdata = mtData.values();
    const size_t *rowIndices = mtData.cols();
    const size_t *colOffsets = mtData.rows();

    Status s;
    size_t partitionNRows = partitionTable->getNumberOfRows();
    Partition<cpu> partitionObj;
    DAAL_CHECK_STATUS(s, partitionObj.init(const_cast<NumericTable *>(partitionTable), fullNUsers));
    const size_t nParts = partitionObj.nParts;
    int *partition = partitionObj.get();

    computeOffsets(nParts, partition, userOffsets);

    /* Split input data table into sub-parts using the partition */
    DAAL_CHECK_STATUS(s, transposeAndSplitCSRTable(nItems, fullNUsers, tdata, rowIndices, colOffsets,
        nParts, partition, dataParts));

    DAAL_CHECK_STATUS(s, computeBlocksToLocal(nItems, fullNUsers, rowIndices, colOffsets,
        nParts, partition, blocksToLocal));

    WriteRows<algorithmFPType, cpu> partialFactors(itemsFactorsTable, 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(partialFactors);
    algorithmFPType *itemsFactors = partialFactors.get();

    /* Initialize item factors */
    size_t nFactors = parameter->nFactors;
    DAAL_CHECK_STATUS(s, computePartialFactors(nUsers, nItems, nFactors, parameter->fullNUsers,
                          tdata, rowIndices, colOffsets, itemsFactors));

    TArray<int, cpu> randBuffer(nFactors);
    DAAL_CHECK_MALLOC(randBuffer.get());

    for (size_t i = 0; i < nItems; i++)
        DAAL_CHECK_STATUS(s, this->randFactors(1, nFactors, itemsFactors + i * nFactors, randBuffer.get(), engine));

    return s;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets(
        size_t nParts, const int *partition, NumericTable **offsets)
{
    for (size_t i = 0; i < nParts; i++)
    {
        WriteRows<int, cpu> offsetRows(offsets[i], 0, 1);
        int *offset = offsetRows.get();
        offset[0] = partition[i];
    }
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::transposeAndSplitCSRTable(size_t nItems, size_t fullNUsers,
            const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets,
            size_t nParts, const int *partition, NumericTable **dataParts)
{
    const size_t nValues = colOffsets[nItems] - colOffsets[0];
    TArray<size_t, cpu> rowOffsetsPtr(fullNUsers + 1);
    TArray<size_t, cpu> colIndicesPtr(nValues);
    TArray<algorithmFPType, cpu> dataPtr(nValues);
    size_t *rowOffsets = rowOffsetsPtr.get();
    size_t *colIndices = colIndicesPtr.get();
    algorithmFPType *data = dataPtr.get();
    DAAL_CHECK_MALLOC(rowOffsets && colIndices && data);
    Status s = training::internal::csr2csc<algorithmFPType, cpu>(fullNUsers, nItems, tdata, rowIndices, colOffsets, data, colIndices, rowOffsets);
    if(!s)
        return s;
    for (size_t i = 0; i < nParts; i++)
    {
        size_t nRowsPart = partition[i + 1] - partition[i];
        size_t nValuesPart = rowOffsets[partition[i + 1]] - rowOffsets[partition[i]];
        CSRNumericTable *dataPartTable = static_cast<CSRNumericTable *>(dataParts[i]);
        DAAL_CHECK_STATUS(s, dataPartTable->allocateDataMemory(nValuesPart));
        WriteRowsCSR<algorithmFPType, cpu> dataPartRows(dataPartTable, 0, nRowsPart);
        DAAL_CHECK_BLOCK_STATUS(dataPartRows);
        size_t *rowOffsetsPart = dataPartRows.rows();
        size_t *colIndicesPart = dataPartRows.cols();
        algorithmFPType *dataPart = dataPartRows.values();

        size_t rowOffsetDiff = rowOffsets[partition[i]] - 1;
        for (size_t j = 0; j < nRowsPart + 1; j++)
        {
            rowOffsetsPart[j] = rowOffsets[j + partition[i]] - rowOffsetDiff;
        }
        size_t offset = rowOffsets[partition[i]] - 1;
        for (size_t j = 0; j < nValuesPart; j++)
        {
            colIndicesPart[j] = colIndices[j + offset];
            dataPart[j]       = data[j + offset];
        }
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal(
            size_t nItems, size_t fullNUsers,
            const size_t *rowIndices, const size_t *colOffsets,
            size_t nParts, const int *partition, NumericTable **blocksToLocal)
{
    TArray<bool, cpu> blockFlagsPtr(nItems * nParts);
    bool *blockFlags = blockFlagsPtr.get();
    DAAL_CHECK_MALLOC(blockFlags);

    for (size_t i = 0; i < nItems; i++)
    {
        for (size_t k = 1; k < nParts + 1; k++)
        {
            blockFlags[(k - 1) * nItems + i] = false;
            for (size_t j = colOffsets[i] - 1; j < colOffsets[i + 1] - 1; j++)
            {
                if (partition[k - 1] <= rowIndices[j] - 1 && rowIndices[j] - 1 < partition[k])
                {
                    blockFlags[(k - 1) * nItems + i] = true;
                }
            }
        }
    }

    TArray<size_t, cpu> blocksToLocalSizePtr(nParts);
    size_t *blocksToLocalSize = blocksToLocalSizePtr.get();
    DAAL_CHECK_MALLOC(blocksToLocalSize);

    for (size_t i = 0; i < nParts; i++)
    {
        blocksToLocalSize[i] = 0;
        for (size_t j = 0; j < nItems; j++)
        {
            blocksToLocalSize[i] += (blockFlags[i * nItems + j] ? 1 : 0);
        }
    }
    for (size_t i = 0; i < nParts; i++)
    {
        size_t nRows = blocksToLocal[i]->getNumberOfRows();
        if ((nRows != 0) && (nRows != blocksToLocalSize[i]))
            return Status(Error::create(ErrorIncorrectNumberOfRows, ArgumentName, outputOfInitForComputeStep3Str()));

        blocksToLocal[i]->resize(blocksToLocalSize[i]);

        WriteRows<int, cpu> blocksToLocalRows(blocksToLocal[i], 0, blocksToLocalSize[i]);
        DAAL_CHECK_BLOCK_STATUS(blocksToLocalRows);
        int *blocksToLocalData = blocksToLocalRows.get();
        size_t indexId = 0;

        for (size_t j = 0; j < nItems; j++)
        {
            if (blockFlags[i * nItems + j])
            {
                blocksToLocalData[indexId++] = (int)j;
            }
        }
    }

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::computePartialFactors(
            size_t nUsers, size_t nItems, size_t nFactors, size_t fullNUsers,
            const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets, algorithmFPType *partialFactors)
{
    const algorithmFPType one(1.0);
    const size_t bufSz = (nItems > nFactors ? nItems : nFactors);

    TArray<algorithmFPType, cpu> onesPtr(nUsers);
    TArray<algorithmFPType, cpu> itemsSumPtr(bufSz);
    algorithmFPType *ones = onesPtr.get();
    algorithmFPType *itemsSum = itemsSumPtr.get();
    DAAL_CHECK_MALLOC(ones && itemsSum);
    service_memset<algorithmFPType, cpu>(ones, one, nUsers);

    /* Parameters of CSRMV function */
    char transa = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    char matdescra[6] = {'\0', '\0', '\0', '\0', '\0', '\0'};
    matdescra[0] = 'G';        // general matrix
    matdescra[3] = 'F';        // 1-based indexing

    /* Compute sum of columns of input matrix */
    SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (DAAL_INT *)&nItems, (DAAL_INT *)&nUsers, &alpha, matdescra,
                        tdata, (DAAL_INT *)rowIndices, (DAAL_INT *)colOffsets, (DAAL_INT *)(colOffsets + 1),
                        ones, &beta, itemsSum);

    const algorithmFPType invFullNUsers = one / algorithmFPType(fullNUsers);
    for (size_t i = 0; i < nItems; i++)
    {
        partialFactors[i * nFactors] = itemsSum[i] * invFullNUsers;
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu>::compute(
            size_t nParts,
            NumericTable **dataParts, NumericTable *dataTable, NumericTable **blocksToLocal,
            NumericTable **itemOffsets)
{
    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();
    CSRNumericTable *csrDataTable = dynamic_cast<CSRNumericTable *>(dataTable);

    size_t nValues = 0;
    for (size_t i = 0; i < nParts; i++)
    {
        nValues += dynamic_cast<CSRNumericTable *>(dataParts[i])->getDataSize();
    }
    Status s;
    DAAL_CHECK_STATUS(s, csrDataTable->allocateDataMemory(nValues));
    WriteRowsCSR<algorithmFPType, cpu> dataTableRows(csrDataTable, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataTableRows);
    algorithmFPType *data = dataTableRows.values();
    size_t *rowOffsets = dataTableRows.rows();
    size_t *colIndices = dataTableRows.cols();
    DAAL_CHECK_STATUS(s, mergeCSRTables(nParts, dataParts, nRows, data, rowOffsets, colIndices));

    TArray<int, cpu> partitionPtr(nParts + 1);
    DAAL_CHECK_MALLOC(partitionPtr.get());
    int *partition = partitionPtr.get();
    partition[0] = 0;
    for (size_t i = 1; i < nParts + 1; i++)
    {
        partition[i] = partition[i - 1] + dataParts[i-1]->getNumberOfColumns();
    }
    computeOffsets(nParts, partition, itemOffsets);
    return computeBlocksToLocal(nRows, nCols, colIndices, rowOffsets, nParts, partition, blocksToLocal);
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu>::mergeCSRTables(
            size_t nParts, NumericTable **dataParts, size_t nRows, algorithmFPType *data,
            size_t *rowOffsets, size_t *colIndices)
{
    TArray<ReadRowsCSR<algorithmFPType, cpu>, cpu> dataPartTables(nParts);
    TArray<const algorithmFPType*, cpu> dataPart(nParts);
    TArray<const size_t*, cpu> rowOffsetsPart(nParts);
    TArray<const size_t*, cpu> colIndicesPart(nParts);
    DAAL_CHECK_MALLOC(dataPartTables.get() && dataPart.get() && rowOffsetsPart.get() && colIndicesPart.get());

    for (size_t p = 0; p < nParts; p++)
    {
        dataPartTables[p].set(dynamic_cast<CSRNumericTableIface *>(dataParts[p]), 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataPartTables[p]);

        dataPart[p] = dataPartTables[p].values();
        rowOffsetsPart[p] = dataPartTables[p].rows();
        colIndicesPart[p] = dataPartTables[p].cols();
    }
    rowOffsets[0] = 1;

    for (size_t i = 1; i < nRows + 1; i++)
    {
        rowOffsets[i] = rowOffsets[i - 1];
        for (size_t p = 0; p < nParts; p++)
        {
            rowOffsets[i] += (rowOffsetsPart[p][i] - rowOffsetsPart[p][i - 1]);
        }
    }

    TArray<size_t, cpu> colIndicesOffsets(nParts);
    DAAL_CHECK_MALLOC(colIndicesOffsets.get());
    colIndicesOffsets[0] = 0;
    for (size_t i = 1; i < nParts; i++)
    {
        colIndicesOffsets[i] = colIndicesOffsets[i - 1] + dataParts[i - 1]->getNumberOfColumns();
    }

    for (size_t i = 1; i < nRows + 1; i++)
    {
        const size_t fullNValues = rowOffsets[i] - rowOffsets[i - 1];
        if(!fullNValues)
            continue;
        TArray<size_t, cpu> colIndicesBufferPtr(fullNValues);
        TArray<algorithmFPType, cpu> dataBufferPtr(fullNValues);
        DAAL_CHECK_MALLOC(colIndicesBufferPtr.get() && dataBufferPtr.get());
        size_t *colIndicesBuffer = colIndicesBufferPtr.get();
        algorithmFPType *dataBuffer = dataBufferPtr.get();
        for (size_t p = 0; p < nParts; p++)
        {
            size_t startCol = rowOffsetsPart[p][i - 1] - 1;
            size_t nValues = rowOffsetsPart[p][i] - rowOffsetsPart[p][i - 1];
            for (size_t j = 0; j < nValues; j++)
            {
                colIndicesBuffer[j] = colIndicesPart[p][startCol + j] + colIndicesOffsets[p];
                dataBuffer[j] = dataPart[p][startCol + j];
            }
            colIndicesBuffer += nValues;
            dataBuffer       += nValues;
        }
        colIndicesBuffer = colIndicesBufferPtr.get();
        dataBuffer = dataBufferPtr.get();
        algorithms::internal::qSort<size_t, algorithmFPType, cpu>(fullNValues, colIndicesBuffer, dataBuffer);
        daal_memcpy_s(colIndices + rowOffsets[i - 1] - 1, fullNValues * sizeof(size_t),
                      colIndicesBuffer,                   fullNValues * sizeof(size_t));
        daal_memcpy_s(data + rowOffsets[i - 1] - 1, fullNValues * sizeof(algorithmFPType),
                      dataBuffer,                   fullNValues * sizeof(algorithmFPType));
    }
    return Status();
}

}
}
}
}
}
}

#endif
