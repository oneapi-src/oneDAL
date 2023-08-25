/* file: outlierdetection_bacon_dense_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of multivariate outlier detection
//--
*/

#ifndef __BACONOUTLIER_DETECTION_DENSE_BACON_IMPL_I__
#define __BACONOUTLIER_DETECTION_DENSE_BACON_IMPL_I__

#include "data_management/data/numeric_table.h"
#include "algorithms/outlier_detection/outlier_detection_bacon_types.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_stat.h"

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
services::Status OutlierDetectionKernel<algorithmFPType, method, cpu>::compute(NumericTable & dataTable, NumericTable & resultTable,
                                                                               const Parameter & par)
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

    const algorithmFPType * data = dataBlock.get();
    algorithmFPType * weight     = resultBlock.get();

    StatisticsInst<algorithmFPType, cpu>::xoutlierdetection(data, (__int64)nFeatures, (__int64)nVectors, nBaconParams, baconParams, weight);

    return Status();
}

} // namespace internal

} // namespace bacon_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
