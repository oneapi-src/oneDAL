/* file: zscore_dense_default_impl.i */
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

#ifndef __ZSCORE_DENSE_DEFAULT_IMPL_I__
#define __ZSCORE_DENSE_DEFAULT_IMPL_I__

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
void ZScoreKernel<algorithmFPType, defaultDense, cpu>::
    computeInternal(NumericTablePtr inputTable, size_t nInputRows, size_t nInputColumns, size_t nBlocks, size_t nRowsInLastBlock,
            daal::algorithms::Parameter *par, algorithmFPType **meanArray, algorithmFPType *standardDeviationInverse)
{
    NumericTablePtr standardDeviationTable;
    BlockDescriptor<algorithmFPType> standardDeviationBlock;
    algorithmFPType *standardDeviationArray;

    Parameter<algorithmFPType, defaultDense> *parameter = static_cast<Parameter<algorithmFPType, defaultDense> *>(par);

    parameter->moments->input.set(low_order_moments::data, inputTable);
    parameter->moments->computeNoThrow();
    if(parameter->moments->getErrors()->size() != 0) {this->_errors->add(ErrorMeanAndStandardDeviationComputing); return;}

    meanTable = parameter->moments->getResult()->get(low_order_moments::mean);
    standardDeviationTable = parameter->moments->getResult()->get(low_order_moments::standardDeviation);
    NumericTablePtr varianceTable = parameter->moments->getResult()->get(low_order_moments::variance);

    meanTable->getBlockOfRows(0, 1, readOnly, meanBlock);
    *meanArray = meanBlock.getBlockPtr();

    standardDeviationTable->getBlockOfRows(0, 1, readOnly, standardDeviationBlock);
    standardDeviationArray = standardDeviationBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> varianceBlock;
    varianceTable->getBlockOfRows(0, 1, readOnly, varianceBlock);
    algorithmFPType *varianceArray = varianceBlock.getBlockPtr();

    for(size_t i = 0; i < nInputColumns; i++)
    {
        if(varianceArray[i] <= zero)
        {
            daal_free(standardDeviationInverse);
            standardDeviationTable->releaseBlockOfRows(standardDeviationBlock);
            varianceTable->releaseBlockOfRows(varianceBlock);
            meanTable->releaseBlockOfRows(meanBlock);

            this->_errors->add(ErrorNullVariance).addIntDetail(Column, (int)i);
            return;
        }
        standardDeviationInverse[i] = one / standardDeviationArray[i];

    }
    varianceTable->releaseBlockOfRows(varianceBlock);
    standardDeviationTable->releaseBlockOfRows(standardDeviationBlock);
}

template<typename algorithmFPType, CpuType cpu>
void ZScoreKernel<algorithmFPType, defaultDense, cpu>::releaseData(algorithmFPType *meanArray)
{
    meanTable->releaseBlockOfRows(meanBlock);
}

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
