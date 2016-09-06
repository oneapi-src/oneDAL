/* file: low_order_moments_online_impl.i */
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
//  Implementation of LowOrderMomentsOnlineKernel
//--
*/

#ifndef __LOW_ORDER_MOMENTS_ONLINE_IMPL_I__
#define __LOW_ORDER_MOMENTS_ONLINE_IMPL_I__

#include "low_order_moments_kernel.h"
#include "low_order_moments_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void LowOrderMomentsOnlineKernel<algorithmFPType, method, cpu>::compute(
            NumericTable *dataTable, PartialResult *partialResult,
            const Parameter *parameter, bool isOnline)
{
    LowOrderMomentsOnlineTask<algorithmFPType, cpu> task(dataTable, partialResult, isOnline, this->_errors);
    if (this->_errors->size() != 0) { return; }

    if (method == sumDense || method == sumCSR)
    {
        retrievePrecomputedStatsIfPossible<algorithmFPType, cpu>(task.nFeatures, task.nVectors,
                dataTable, task.resultArray[(int)partialSum], task.mean, this->_errors);
        if (this->_errors->size() != 0) { return; }
    }

    computeSumAndVariance<algorithmFPType, method, cpu>(task.nFeatures, task.nVectors, task.dataBlock,
        task.resultArray[(int)partialSum], task.prevSums, task.mean, task.raw2Mom, task.variance,
        isOnline, this->_errors);
    if (this->_errors->size() != 0) { return; }

    if (!isOnline)
    {
        initializeMinAndMax<algorithmFPType, cpu>(task.nFeatures, task.dataBlock,
            task.resultArray[(int)partialMinimum], task.resultArray[(int)partialMaximum]);
    }

    computeMinAndMax<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock,
        task.resultArray[(int)partialMinimum], task.resultArray[(int)partialMaximum]);

    computeSumOfSquares<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock,
        task.resultArray[(int)partialSumSquares], isOnline);

    computeSumOfSquaredDiffsFromMean<algorithmFPType, cpu>(task.nFeatures, task.nVectors,
        (size_t)(task.resultArray[(int)nObservations][0]),
        task.variance, task.resultArray[(int)partialSum], task.prevSums,
        task.resultArray[(int)partialSumSquaresCentered], isOnline);

    task.resultArray[(int)nObservations][0] += (algorithmFPType)(task.nVectors);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LowOrderMomentsOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
            NumericTable *nObservationsTable,
            NumericTable *sumTable, NumericTable *sumSqTable, NumericTable *sumSqCenTable,
            NumericTable *meanTable, NumericTable *raw2MomTable, NumericTable *varianceTable,
            NumericTable *stDevTable, NumericTable *variationTable,
            const Parameter *parameter)
{
    LowOrderMomentsFinalizeTask<algorithmFPType, cpu> task(
        nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable,
        raw2MomTable, varianceTable, stDevTable, variationTable);

    finalize<algorithmFPType, cpu>(task);
}

}
}
}
}

#endif
