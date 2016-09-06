/* file: zscore_dense_sum_impl.i */
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
//  Implementation of sumDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_SUM_IMPL_I__
#define __ZSCORE_DENSE_SUM_IMPL_I__

#include "service_micro_table.h"
#include "service_math.h"

using namespace daal::internal;
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
void ZScoreKernel<algorithmFPType, sumDense, cpu>::
    computeInternal(NumericTablePtr inputTable, size_t nInputRows, size_t nInputColumns, size_t nBlocks, size_t nRowsInLastBlock,
            daal::algorithms::Parameter *par, algorithmFPType **meanArray, algorithmFPType *standardDeviationInverse)
{
    algorithmFPType *standardDeviationArray;

    NumericTablePtr sumTable = inputTable->basicStatistics.get(NumericTableIface::sum);
    if(sumTable.get() == 0)
    {
        daal_free(standardDeviationInverse);
        this->_errors->add(services::ErrorPrecomputedSumNotAvailable);
        return;
    }

    BlockDescriptor<algorithmFPType> sumBlock;
    sumTable->getBlockOfRows(0, 1, readOnly, sumBlock);
    algorithmFPType *sumArray = sumBlock.getBlockPtr();

    *meanArray = (algorithmFPType *) daal_malloc(nInputColumns * sizeof(algorithmFPType));
    if (!meanArray)
    {
        daal_free(standardDeviationInverse);
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    standardDeviationArray = (algorithmFPType *) daal_malloc(nInputColumns * sizeof(algorithmFPType));
    if (!standardDeviationArray)
    {
        daal_free(standardDeviationInverse);
        daal_free(*meanArray);
        sumTable->releaseBlockOfRows(sumBlock);
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    algorithmFPType rowsInverseValue = one / (algorithmFPType)nInputRows;
    algorithmFPType inverseValue = one / ((algorithmFPType)nInputRows - one);

    for(size_t i = 0; i < nInputColumns; i++)
    {
        (*meanArray)[i] = sumArray[i] * rowsInverseValue;
        standardDeviationArray[i] = zero;
    }

    for(size_t block = 0; block < nBlocks; block++)
    {
        getSumSquaresCentered(inputTable, nInputColumns, block * this->_nRowsInBlock, this->_nRowsInBlock, standardDeviationArray, *meanArray);
    }
    if(nRowsInLastBlock > 0)
    {
        getSumSquaresCentered(inputTable, nInputColumns, nBlocks * this->_nRowsInBlock, nRowsInLastBlock, standardDeviationArray, *meanArray);
    }

    algorithmFPType value = daal::internal::Math<algorithmFPType,cpu>::sSqrt((algorithmFPType)nInputRows - one);
    daal::internal::Math<algorithmFPType,cpu>::vSqrt(nInputColumns, standardDeviationArray, standardDeviationArray);

    for(size_t i = 0; i < nInputColumns; i++)
    {
        if(standardDeviationArray[i] <= zero)
        {
            daal_free(standardDeviationInverse);
            daal_free(*meanArray);
            daal_free(standardDeviationArray);
            sumTable->releaseBlockOfRows(sumBlock);
            this->_errors->add(ErrorNullVariance).addIntDetail(Column, (int)i);
            return;
        }
        standardDeviationInverse[i] = value / standardDeviationArray[i];
    }

    sumTable->releaseBlockOfRows(sumBlock);
    daal_free(standardDeviationArray);
}

template<typename algorithmFPType, CpuType cpu>
void ZScoreKernel<algorithmFPType, sumDense, cpu>::
    getSumSquaresCentered(NumericTablePtr inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                          algorithmFPType *standardDeviationArray, algorithmFPType *meanArray)
{
    BlockDescriptor<algorithmFPType> inputBlock;
    inputTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockPtr();

    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        for(size_t j = 0; j < nInputColumns; j++)
        {
            standardDeviationArray[j] += (inputArray[i * nInputColumns + j] - meanArray[j]) * (inputArray[i * nInputColumns + j] - meanArray[j]);
        }
    }
    inputTable->releaseBlockOfRows(inputBlock);
}

template<typename algorithmFPType, CpuType cpu>
void ZScoreKernel<algorithmFPType, sumDense, cpu>::releaseData(algorithmFPType *meanArray)
{
    daal_free(meanArray);
}

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
