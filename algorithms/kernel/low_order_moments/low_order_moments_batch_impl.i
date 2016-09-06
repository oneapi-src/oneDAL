/* file: low_order_moments_batch_impl.i */
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
//  Implementation of LowOrderMomentsBatchKernel
//--
*/

#ifndef __LOW_ORDER_MOMENTS_BATCH_IMPL_I__
#define __LOW_ORDER_MOMENTS_BATCH_IMPL_I__

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
void LowOrderMomentsBatchKernel<algorithmFPType, method, cpu>::compute(
            NumericTable *dataTable, Result *result, const Parameter *parameter)
{
    bool isOnline = false;
    LowOrderMomentsBatchTask<algorithmFPType, cpu> task(dataTable, result);

    if (method == sumDense || method == sumCSR)
    {
        retrievePrecomputedStatsIfPossible<algorithmFPType, cpu>(task.nFeatures, task.nVectors,
                dataTable, task.resultArray[(int)sum], task.resultArray[(int)mean], this->_errors);
        if (this->_errors->size() != 0) { return; }
    }

    computeSum_Mean_SecondOrderRawMoment_Variance_Variation<algorithmFPType, method, cpu>(
        task.nFeatures, task.nVectors, task.dataBlock, task.resultArray[(int)sum],
        task.resultArray[(int)mean], task.resultArray[(int)secondOrderRawMoment],
        task.resultArray[(int)variance], task.resultArray[(int)variation], this->_errors);
    if (this->_errors->size() != 0) { return; }

    /* Compute standard deviation */
    daal::internal::Math<algorithmFPType,cpu>::vSqrt(task.nFeatures, task.resultArray[(int)variance], task.resultArray[(int)standardDeviation]);

    initializeMinAndMax<algorithmFPType, cpu>(task.nFeatures, task.dataBlock,
        task.resultArray[(int)minimum], task.resultArray[(int)maximum]);

    computeMinAndMax<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock,
        task.resultArray[(int)minimum], task.resultArray[(int)maximum]);

    computeSumOfSquares<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock,
        task.resultArray[(int)sumSquares], isOnline);

    computeSumOfSquaredDiffsFromMean<algorithmFPType, cpu>(task.nFeatures, task.nVectors, 0,
        task.resultArray[(int)variance], task.resultArray[(int)sum], task.resultArray[(int)sum],
        task.resultArray[(int)sumSquaresCentered], isOnline);
}

}
}
}
}

#endif
