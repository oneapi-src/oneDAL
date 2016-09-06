/* file: outlierdetection_multivariate_dense_bacon_impl.i */
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
//  Implementation of multivariate outlier detection
//--
*/

#ifndef __MULTIVARIATE_OUTLIER_DETECTION_DENSE_BACON_IMPL_I__
#define __MULTIVARIATE_OUTLIER_DETECTION_DENSE_BACON_IMPL_I__

#include "numeric_table.h"
#include "outlier_detection_multivariate_types.h"

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_stat.h"

#include "outlierdetection_multivariate_dense_bacon_kernel.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
void OutlierDetectionKernel<AlgorithmFPType, baconDense, cpu>::compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par)
{
    const __int64 nBaconParams = 3;
    AlgorithmFPType baconParams[nBaconParams];
    Parameter<baconDense> *odPar = static_cast<Parameter<baconDense> *>(const_cast<daal::algorithms::Parameter *>(par));
    if (odPar == NULL)
    {
        odPar = new Parameter<baconDense>();
    }

    switch (odPar->initMethod)
    {
    case baconMedian:
    default:
    {
        baconParams[0] = (AlgorithmFPType)__DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT;
        break;
    }
    case baconMahalanobis:
    {
        baconParams[0] = (AlgorithmFPType)__DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT;
        break;
    }
    }
    baconParams[1] = (AlgorithmFPType)(odPar->alpha);
    baconParams[2] = (AlgorithmFPType)(odPar->toleranceToConverge);

    BlockMicroTable<AlgorithmFPType, readOnly, cpu> mtA(a);
    FeatureMicroTable<AlgorithmFPType, writeOnly, cpu> mtR(r);
    size_t nFeatures = mtA.getFullNumberOfColumns();
    size_t nVectors  = mtA.getFullNumberOfRows();

    AlgorithmFPType *data, *weight;
    mtA.getBlockOfRows(0, nVectors, &data);
    mtR.getBlockOfColumnValues(0, 0, nVectors, &weight);

    Statistics<AlgorithmFPType, cpu>::xoutlierdetection(data, (__int64)nFeatures, (__int64)nVectors, nBaconParams, baconParams, weight);

    mtA.release();
    mtR.release();
    if (par == NULL) { delete odPar; }
}

} // namespace internal

} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
