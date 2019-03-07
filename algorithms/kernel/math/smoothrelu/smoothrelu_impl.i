/* file: smoothrelu_impl.i */
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
