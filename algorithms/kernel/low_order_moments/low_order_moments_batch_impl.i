/* file: low_order_moments_batch_impl.i */
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
services::Status LowOrderMomentsBatchKernel<algorithmFPType, method, cpu>::compute( NumericTable *dataTable,
                                                                                    Result *result,
                                                                                    const Parameter *parameter )
{
    if( method == defaultDense)
    {
        switch(parameter->estimatesToCompute)
        {
            case estimatesMinMax:
                return estimates_batch_minmax::compute_estimates<algorithmFPType,cpu>(dataTable, result);
            case estimatesMeanVariance:
                return estimates_batch_meanvariance::compute_estimates<algorithmFPType,cpu>(dataTable, result);
            default /* estimatesAll */:
                break;
        }
        return estimates_batch_all::compute_estimates<algorithmFPType, cpu>(dataTable, result);
    }

    LowOrderMomentsBatchTask<algorithmFPType, cpu> task(dataTable, result);


    if (method == sumDense || method == sumCSR)
    {
        Status s = retrievePrecomputedStatsIfPossible<algorithmFPType, cpu>(task.nFeatures,
                                                                            task.nVectors,
                                                                            dataTable,
                                                                            task.resultArray[(int)sum],
                                                                            task.resultArray[(int)mean]);

        if(!s)
            return s;

    }
    Status s = computeSum_Mean_SecondOrderRawMoment_Variance_Variation<algorithmFPType, method, cpu>(task.nFeatures,
                                                                                                     task.nVectors,
                                                                                                     task.dataBlock,
                                                                                                     task.resultArray[(int)sum],
                                                                                                     task.resultArray[(int)mean],
                                                                                                     task.resultArray[(int)secondOrderRawMoment],
                                                                                                     task.resultArray[(int)variance],
                                                                                                     task.resultArray[(int)variation]);


    if(!s)
        return s;

    const bool isOnline = false;

    /* Compute standard deviation */
    daal::internal::Math<algorithmFPType,cpu>::vSqrt( task.nFeatures,
                                                      task.resultArray[(int)variance],
                                                      task.resultArray[(int)standardDeviation] );

    computeMinMaxAndSumOfSquared<algorithmFPType, cpu> (task.nFeatures,
                                            task.nVectors,
                                            task.dataBlock,
                                            task.resultArray[(int)minimum],
                                            task.resultArray[(int)maximum],
                                            task.resultArray[(int)sumSquares],
                                            isOnline);


    computeSumOfSquaredDiffsFromMean<algorithmFPType, cpu>( task.nFeatures,
                                                            task.nVectors,
                                                            0,
                                                            task.resultArray[(int)variance],
                                                            task.resultArray[(int)sum],
                                                            task.resultArray[(int)sum],
                                                            task.resultArray[(int)sumSquaresCentered],
                                                            isOnline);

    return s;
}

}
}
}
}

#endif
