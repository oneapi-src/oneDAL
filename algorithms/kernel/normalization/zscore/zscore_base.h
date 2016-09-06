/* file: zscore_base.h */
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
//  Declaration of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_BASE_H__
#define __ZSCORE_BASE_H__

#include "normalization/zscore.h"
#include "kernel.h"
#include "numeric_table.h"

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
/**
 *  \brief Kernel for zscore normalization calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, CpuType cpu>
class ZScoreKernelBase : public Kernel
{
public:

    /**
     *  \brief Function that computes z-score normalization
     *
     *  \param input[in]        Input of the algorithm
     *  \param result[out]      Result of the algorithm
     *  \param parameter[in]    Parameters of the algorithm
     */
    void compute(const Input *input, Result *result, daal::algorithms::Parameter *parameter);

    virtual void computeInternal(NumericTablePtr inputTable, size_t nInputRows, size_t nInputColumns, size_t nBlocks, size_t nRowsInLastBlock,
                         daal::algorithms::Parameter *parameter, algorithmFPType **meanArray, algorithmFPType *standardDeviationInverse) = 0;

    virtual void releaseData(algorithmFPType *meanArray) = 0;

protected:
    const size_t _nRowsInBlock = 5000;

private:

    /**
     *  \brief Checks for inplace normalization
     *
     *  \param  inputTable[in]        Numeric table containing input data
     *  \param  resultTable[out]      Numeric table containing normalization results
     *  \param  nInputColumns[in]     Number of columns in input table
     *  \param  nBlocks[in]           Number of data table
     *  \param  nRowsInLastBlock[in]  Number of rows in the last block of data
     */
    inline void checkForInplace(NumericTablePtr inputTable, NumericTablePtr resultTable, size_t nInputColumns, size_t nBlocks,
                                size_t nRowsInLastBlock);

    /**
     *  \brief Copies data from input table to result table if data is already normalized
     *
     *  \param  inputTable[in]          Numeric table containing input data
     *  \param  nInputColumns[in]       Number of columns in input table
     *  \param  nProcessedRows[in]      Number of processed rows
     *  \param  nRowsInCurrentBlock[in] Number of rows to process
     *  \param  resultTable[out]        Numeric table containing normalization results
     */
    inline void copyDataBlock(NumericTablePtr inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                              NumericTablePtr resultTable);

    /**
     *  \brief Normalizes data from input table by blocks
     *
     *  \param  inputTable[in]               Numeric table containing input data
     *  \param  nInputColumns[in]            Number of columns in input table
     *  \param  nProcessedRows[in]           Number of processed rows
     *  \param  nRowsInCurrentBlock[in]      Number of rows to process
     *  \param  resultTable[out]             Numeric table containing normalization results
     *  \param  meanArray[in]                Array of mean values
     *  \param  standardDeviationInverse[in] Array of inversed values of standard deviations
     */
    inline void normalizeDataInBlock(NumericTablePtr inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                     NumericTablePtr resultTable, algorithmFPType *meanArray, algorithmFPType *standardDeviationInverse);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ZScoreKernel : public ZScoreKernelBase<algorithmFPType, cpu>
{};

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#include "zscore_impl.i"

#endif
