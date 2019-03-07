/* file: stump_train_aux.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_TRAIN_AUX_I__
#define __STUMP_TRAIN_AUX_I__

#include "daal_defines.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace internal
{
using namespace daal::internal;

/**
 *  \brief Perform stump regression for data set X on responses Y with weights W
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::compute(size_t n, const NumericTable *const *a, stump::Model *r,
                                                             const Parameter *par)
{
    const NumericTable *xTable = a[0];
    const NumericTable *yTable = a[1];
    const NumericTable *wTable = (n >= 3 ? a[2] : 0);

    const size_t nFeatures = xTable->getNumberOfColumns();
    const size_t nVectors  = xTable->getNumberOfRows();
    r->setNFeatures(nFeatures);

    services::Status s;
    ReadColumns<algorithmFPtype, cpu> wBlock(const_cast<NumericTable *>(wTable), 0, 0, nVectors);
    TArray<algorithmFPtype, cpu> wArray(wTable ? 0 : nVectors);
    if(wTable)
    {
        /* Here if array of weight is provided */
        DAAL_CHECK_STATUS(s, wBlock.status());
    }
    else
    {
        DAAL_CHECK(wArray.get(), services::ErrorMemoryAllocationFailed);
        /* Here if array of weight is not provided */
        const algorithmFPtype weight = 1.0 / algorithmFPtype(nVectors);
        algorithmFPtype *w = wArray.get();
        for(size_t i = 0; i < nVectors; i++)
            w[i] = weight;
        }

    algorithmFPtype splitPoint;
    algorithmFPtype leftValue;
    algorithmFPtype rightValue;
    size_t splitFeature;
    {
        ReadColumns<algorithmFPtype, cpu> y(const_cast<NumericTable *>(yTable), 0, 0, nVectors);
        DAAL_CHECK_STATUS(s, y.status());
        doStumpRegression(nVectors, nFeatures, xTable, (wTable ? wBlock.get() : wArray.get()), y.get(),
            splitFeature, splitPoint, leftValue, rightValue);
    }

    r->setSplitFeature(splitFeature);
    r->setSplitValue<algorithmFPtype>(splitPoint);
    r->setLeftSubsetAverage<algorithmFPtype>(leftValue);
    r->setRightSubsetAverage<algorithmFPtype>(rightValue);

    return s;
}

} // namespace daal::algorithms::stump::training::internal
}
}
}
} // namespace daal

#endif
