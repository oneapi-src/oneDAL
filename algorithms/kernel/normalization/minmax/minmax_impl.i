/* file: minmax_impl.i */
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
//  Implementation of minmax algorithm
//--
*/

#ifndef __MINMAX_IMPL_I__
#define __MINMAX_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace internal
{

/**
 *  \brief Kernel for min-max calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status MinMaxKernel<algorithmFPType, method, cpu>::compute(const NumericTable &inputTable, NumericTable &resultTable,
                                                           const NumericTable &minimums, const NumericTable &maximums,
                                                           const algorithmFPType lowerBound, const algorithmFPType upperBound)
{
    ReadRows<algorithmFPType, cpu, NumericTable> minimumsTableRows(const_cast<NumericTable &>(minimums), 0, minimums.getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(minimumsTableRows);
    ReadRows<algorithmFPType, cpu, NumericTable> maximumsTableRows(const_cast<NumericTable &>(maximums), 0, maximums.getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(maximumsTableRows);

    const algorithmFPType *minArray = minimumsTableRows.get();
    const algorithmFPType *maxArray = maximumsTableRows.get();

    const size_t nRows = inputTable.getNumberOfRows();
    const size_t nColumns = inputTable.getNumberOfColumns();

    TArray<algorithmFPType, cpu> scaleFactorsPtr(nColumns);
    TArray<algorithmFPType, cpu> shiftFactorsPtr(nColumns);

    algorithmFPType *scaleFactors = scaleFactorsPtr.get();
    algorithmFPType *shiftFactors = shiftFactorsPtr.get();
    DAAL_CHECK(scaleFactors && shiftFactors, ErrorMemoryAllocationFailed);

    algorithmFPType delta = upperBound - lowerBound;
    for(size_t j = 0; j < nColumns; j++)
    {
        scaleFactors[j] = delta / (maxArray[j] - minArray[j]);
        shiftFactors[j] = minArray[j] * scaleFactors[j] - lowerBound;
    }

    size_t regularBlockSize = (nRows > BLOCK_SIZE_NORM) ? BLOCK_SIZE_NORM : nRows;
    size_t blocksNumber = nRows / regularBlockSize;

    SafeStatus safeStat;
    daal::threader_for(blocksNumber, blocksNumber, [ & ](int iRowsBlock)
    {
        size_t blockSize = regularBlockSize;
        size_t startRowIndex = iRowsBlock * regularBlockSize;

        if(iRowsBlock == blocksNumber - 1)
        {
            blockSize += nRows % regularBlockSize;
        }

        safeStat |= processBlock(inputTable, resultTable, scaleFactors, shiftFactors, startRowIndex, blockSize);
    } );

    resultTable.setNormalizationFlag(NumericTableIface::minMaxNormalized);
    return safeStat.detach();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status MinMaxKernel<algorithmFPType, method, cpu>::processBlock(const NumericTable &inputTable, NumericTable &resultTable,
                                                                const algorithmFPType *scale, const algorithmFPType *shift,
                                                                const size_t startRowIndex, const size_t blockSize)
{
    const size_t nColumns = inputTable.getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> inputTableRows(const_cast<NumericTable &>(inputTable), startRowIndex, blockSize);
    DAAL_CHECK_BLOCK_STATUS(inputTableRows);
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> resultTableRows(resultTable, startRowIndex, blockSize);
    DAAL_CHECK_BLOCK_STATUS(resultTableRows);

    const algorithmFPType *input = inputTableRows.get();
    algorithmFPType *result = resultTableRows.get();

    for(size_t i = 0; i < blockSize; i++)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nColumns; j++)
        {
            result[i * nColumns + j] = input[i * nColumns + j] * scale[j] - shift[j];
        }
    }
    return Status();
}

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
