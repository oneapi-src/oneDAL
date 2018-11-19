/* file: low_order_moments_kernel.h */
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
//  Declaration of template function that calculate low order moments.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_KERNEL_H__
#define __LOW_ORDER_MOMENTS_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "low_order_moments_types.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{

template<typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, Result *result, const Parameter *parameter);
};

template<typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsOnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, PartialResult *partialResult,
            const Parameter *parameter, bool isOnline);

    services::Status finalizeCompute(NumericTable *nObservationsTable,
            NumericTable *sumTable, NumericTable *sumSqTable, NumericTable *sumSqCenTable,
            NumericTable *meanTable, NumericTable *raw2MomTable, NumericTable *varianceTable,
            NumericTable *stDevTable, NumericTable *variationTable,
            const Parameter *parameter);
};

template<typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsDistributedKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(data_management::DataCollection *partialResultsCollection,
            PartialResult *partialResult, const Parameter *parameter);

    services::Status finalizeCompute(NumericTable *nObservationsTable,
            NumericTable *sumTable, NumericTable *sumSqTable, NumericTable *sumSqCenTable,
            NumericTable *meanTable, NumericTable *raw2MomTable, NumericTable *varianceTable,
            NumericTable *stDevTable, NumericTable *variationTable,
            const Parameter *parameter);
};


template<typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsBatchTask
{
    LowOrderMomentsBatchTask(NumericTable *dataTable, Result *result);
    virtual ~LowOrderMomentsBatchTask();

    size_t nVectors;
    size_t nFeatures;

    NumericTable *dataTable;
    NumericTablePtr resultTable[lastResultId + 1];

    BlockDescriptor<algorithmFPType> dataBD;
    BlockDescriptor<algorithmFPType> resultBD[lastResultId + 1];

    algorithmFPType *dataBlock;
    algorithmFPType *resultArray[lastResultId + 1];
};

template<typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsOnlineTask
{
    LowOrderMomentsOnlineTask(NumericTable *dataTable);
    virtual ~LowOrderMomentsOnlineTask();
    Status init(PartialResult *partialResult, bool isOnline);

    size_t nVectors;
    size_t nFeatures;

    NumericTable *dataTable;
    NumericTablePtr resultTable[lastPartialResultId + 1];

    BlockDescriptor<algorithmFPType> dataBD;
    BlockDescriptor<algorithmFPType> resultBD[lastPartialResultId + 1];

    algorithmFPType *dataBlock;
    algorithmFPType *resultArray[lastPartialResultId + 1];

    algorithmFPType *mean;
    algorithmFPType *raw2Mom;
    algorithmFPType *variance;
    algorithmFPType *stDev;
    algorithmFPType *variation;
    algorithmFPType *prevSums;
};

template<typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsFinalizeTask
{
    LowOrderMomentsFinalizeTask(NumericTable *nObservationsTable,
            NumericTable *sumTable, NumericTable *sumSqTable, NumericTable *sumSqCenTable,
            NumericTable *meanTable, NumericTable *raw2MomTable, NumericTable *varianceTable,
            NumericTable *stDevTable, NumericTable *variationTable);
    virtual ~LowOrderMomentsFinalizeTask();

    size_t nFeatures;

    NumericTable *nObservationsTable;
    NumericTable *sumTable;
    NumericTable *sumSqTable;
    NumericTable *sumSqCenTable;

    NumericTable *meanTable;
    NumericTable *raw2MomTable;
    NumericTable *varianceTable;
    NumericTable *stDevTable;
    NumericTable *variationTable;

    BlockDescriptor<int> nObservationsBD;
    BlockDescriptor<algorithmFPType> sumBD;
    BlockDescriptor<algorithmFPType> sumSqBD;
    BlockDescriptor<algorithmFPType> sumSqCenBD;

    BlockDescriptor<algorithmFPType> meanBD;
    BlockDescriptor<algorithmFPType> raw2MomBD;
    BlockDescriptor<algorithmFPType> varianceBD;
    BlockDescriptor<algorithmFPType> stDevBD;
    BlockDescriptor<algorithmFPType> variationBD;

    int *nObservations;
    algorithmFPType *sums;
    algorithmFPType *sumSq;
    algorithmFPType *sumSqCen;

    algorithmFPType *mean;
    algorithmFPType *raw2Mom;
    algorithmFPType *variance;
    algorithmFPType *stDev;
    algorithmFPType *variation;
};

}
}
}
}

#endif
