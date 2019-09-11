/* file: outlierdetection_bacon_dense_impl.i */
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
//  Implementation of multivariate outlier detection
//--
*/

#ifndef __BACONOUTLIER_DETECTION_DENSE_BACON_IMPL_I__
#define __BACONOUTLIER_DETECTION_DENSE_BACON_IMPL_I__

#include "numeric_table.h"
#include "outlier_detection_bacon_types.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_stat.h"

namespace daal
{
namespace algorithms
{
namespace bacon_outlier_detection
{
namespace internal
{

using namespace daal::internal;
using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status OutlierDetectionKernel<algorithmFPType, method, cpu>::compute(NumericTable &dataTable, NumericTable &resultTable, const Parameter &par)
{
    const __int64 nBaconParams = 3;
    algorithmFPType baconParams[nBaconParams];

    switch (par.initMethod)
    {
    case baconMedian:
    default:
    {
        baconParams[0] = (algorithmFPType)__DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT;
        break;
    }
    case baconMahalanobis:
    {
        baconParams[0] = (algorithmFPType)__DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT;
        break;
    }
    }
    baconParams[1] = (algorithmFPType)(par.alpha);
    baconParams[2] = (algorithmFPType)(par.toleranceToConverge);

    size_t nFeatures = dataTable.getNumberOfColumns();
    size_t nVectors  = dataTable.getNumberOfRows();

    ReadRows<algorithmFPType, cpu> dataBlock(dataTable, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(dataBlock)

    WriteOnlyRows<algorithmFPType, cpu> resultBlock(resultTable, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(resultBlock)

    const algorithmFPType *data = dataBlock.get();
    algorithmFPType *weight = resultBlock.get();

    Statistics<algorithmFPType, cpu>::xoutlierdetection(data, (__int64)nFeatures, (__int64)nVectors, nBaconParams, baconParams, weight);

    return Status();
}

} // namespace internal

} // namespace bacon_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
