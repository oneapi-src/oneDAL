/* file: zscore_dense_default_kernel.h */
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
//  Implementation of defaultDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_DEFAULT_KERNEL_H__
#define __ZSCORE_DENSE_DEFAULT_KERNEL_H__

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

/**
 *  \brief Specialization of the structure that contains kernels for z-score normalization using defaultDense method
 */
template<typename algorithmFPType, CpuType cpu>
class ZScoreKernel<algorithmFPType, defaultDense, cpu> : public ZScoreKernelBase<algorithmFPType, cpu>
{
public:
//    using ZScoreKernelBase<algorithmFPType, cpu>::compute;

    algorithmFPType zero = 0.0;
    algorithmFPType one  = 1.0;

    /**
     *  \brief Specialization of compute() function
     *
     *  \param  inputTable[in]               Numeric table containing input data
     *  \param  nInputRows[in]               Number of rows in input table, is used in sumDense method
     *  \param  nInputColumns[in]            Number of columns in input table
     *  \param  nBlocks[in]                  Total number of data blocks, is used in sumDense method
     *  \param  nRowsInLastBlock[in]         Number of rows in last block, is used in sumDense method
     *  \param  par[in]                      Parameters of the algorithm
     *  \param  meanArray[out]               Array of mean values
     *  \param  standardDeviationInverse[in] Array of inversed values of standard deviations
     */
    void computeInternal(NumericTablePtr inputTable, size_t nInputRows, size_t nInputColumns, size_t nBlocks, size_t nRowsInLastBlock,
                 daal::algorithms::Parameter *par, algorithmFPType **meanArray, algorithmFPType *standardDeviationInverse);

    /**
     *  \brief Releases used blocks
     *  \param  meanArray[in]   Array containing means of input data, needed to be released in sumDense method
     */
    void releaseData(algorithmFPType *meanArray);

private:
    NumericTablePtr meanTable;
    BlockDescriptor<algorithmFPType> meanBlock;
};

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
