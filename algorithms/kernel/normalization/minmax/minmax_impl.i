/* file: minmax_impl.i */
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
//  Implementation of minmax algorithm
//--
*/

#ifndef __MINMAX_IMPL_I__
#define __MINMAX_IMPL_I__

#include "threading.h"
#include "service_memory.h"
#include "service_numeric_table.h"

using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::data_management;

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
void MinMaxKernel<algorithmFPType, method, cpu>::compute(NumericTable *inputTable, NumericTable *resultTable,
                                                         NumericTable *minimums, NumericTable *maximums,
                                                         algorithmFPType lowerBound, algorithmFPType upperBound)
{
    daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> minimumsTableRows(minimums, 0, minimums->getNumberOfRows());
    daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> maximumsTableRows(maximums, 0, maximums->getNumberOfRows());
    const algorithmFPType *minArray = minimumsTableRows.get();
    const algorithmFPType *maxArray = maximumsTableRows.get();

    const size_t nRows = inputTable->getNumberOfRows();
    const size_t nColumns = inputTable->getNumberOfColumns();

    daal::internal::TSmartPtr<algorithmFPType, cpu> scaleFactorsPtr(nColumns);
    daal::internal::TSmartPtr<algorithmFPType, cpu> shiftFactorsPtr(nColumns);

    algorithmFPType *scaleFactors = scaleFactorsPtr.get();
    algorithmFPType *shiftFactors = shiftFactorsPtr.get();
    if(!scaleFactors || !shiftFactors)
    {
        this->_errors->add(daal::services::ErrorMemoryAllocationFailed);
        return;
    }

    algorithmFPType delta = upperBound - lowerBound;
    for(size_t j = 0; j < nColumns; j++)
    {
        scaleFactors[j] = delta / (maxArray[j] - minArray[j]);
        shiftFactors[j] = minArray[j] * scaleFactors[j] - lowerBound;
    }

    size_t regularBlockSize = (nRows > BLOCK_SIZE_NORM) ? BLOCK_SIZE_NORM : nRows;
    size_t blocksNumber = nRows / regularBlockSize;

    daal::threader_for(blocksNumber, blocksNumber, [ & ](int iRowsBlock)
    {
        size_t blockSize = regularBlockSize;
        size_t startRowIndex = iRowsBlock * regularBlockSize;

        if(iRowsBlock == blocksNumber - 1)
        {
            blockSize += nRows % regularBlockSize;
        }

        processBlock(inputTable, resultTable, scaleFactors, shiftFactors, startRowIndex, blockSize);
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
void MinMaxKernel<algorithmFPType, method, cpu>::processBlock(NumericTable *inputTable, NumericTable *resultTable,
                                                              algorithmFPType *scale, algorithmFPType *shift,
                                                              size_t startRowIndex, size_t blockSize)
{
    const size_t nColumns = inputTable->getNumberOfColumns();

    daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> inputTableRows(inputTable, startRowIndex, blockSize);
    daal::internal::WriteOnlyRows<algorithmFPType, cpu, NumericTable> resultTableRows(resultTable, startRowIndex, blockSize);

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
}

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
