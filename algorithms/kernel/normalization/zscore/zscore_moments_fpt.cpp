/* file: zscore_moments_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "zscore_moments.h"

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::low_order_moments;
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
template <typename algorithmFPType>
services::Status computeMeansAndVariances(daal::algorithms::low_order_moments::BatchImpl * moments,
                                          const daal::data_management::NumericTable & dataTable, algorithmFPType * resultMean,
                                          algorithmFPType * resultVariance)
{
    moments->parameter.estimatesToCompute = low_order_moments::estimatesMeanVariance;
    const size_t _nFeatures               = dataTable.getNumberOfColumns();
    auto meansPtr                         = HomogenNumericTable<algorithmFPType>::create(resultMean, _nFeatures, 1);
    auto variancesPtr                     = HomogenNumericTable<algorithmFPType>::create(resultVariance, _nFeatures, 1);

    low_order_moments::ResultPtr meanVarianceResult(new low_order_moments::Result());
    meanVarianceResult->allocate<algorithmFPType>(&(moments->input), &(moments->parameter), defaultDense);
    meanVarianceResult->set(low_order_moments::mean, meansPtr);
    meanVarianceResult->set(low_order_moments::variance, variancesPtr);
    moments->setResult(meanVarianceResult);

    DAAL_CHECK(moments->computeNoThrow(), ErrorMeanAndStandardDeviationComputing);

    return Status();
}

template services::Status computeMeansAndVariances<DAAL_FPTYPE>(low_order_moments::BatchImpl * moments,
                                                                const daal::data_management::NumericTable & dataTable, DAAL_FPTYPE * resultMean,
                                                                DAAL_FPTYPE * resultVariance);
} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
