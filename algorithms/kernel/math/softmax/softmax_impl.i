/* file: softmax_impl.i */
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
