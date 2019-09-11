/* file: zscore_moments_fpt.cpp */
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
template<typename algorithmFPType>
services::Status computeMeansAndVariances(daal::algorithms::low_order_moments::BatchImpl *moments,
                                          const daal::data_management::NumericTable &dataTable,
                                          algorithmFPType* resultMean,
                                          algorithmFPType* resultVariance)
{
    moments->parameter.estimatesToCompute = low_order_moments::estimatesMeanVariance;
    const size_t _nFeatures = dataTable.getNumberOfColumns();
    auto meansPtr = HomogenNumericTable<algorithmFPType>::create(resultMean, _nFeatures, 1);
    auto variancesPtr = HomogenNumericTable<algorithmFPType>::create(resultVariance, _nFeatures, 1);

    low_order_moments::ResultPtr meanVarianceResult(new low_order_moments::Result());
    meanVarianceResult->allocate<algorithmFPType>(&(moments->input), &(moments->parameter), defaultDense);
    meanVarianceResult->set(low_order_moments::mean, meansPtr);
    meanVarianceResult->set(low_order_moments::variance, variancesPtr);
    moments->setResult(meanVarianceResult);

    DAAL_CHECK(moments->computeNoThrow(), ErrorMeanAndStandardDeviationComputing);

    return Status();
}


template
services::Status computeMeansAndVariances<DAAL_FPTYPE>(low_order_moments::BatchImpl *moments,
                                                       const daal::data_management::NumericTable &dataTable,
                                                       DAAL_FPTYPE* resultMean,
                                                       DAAL_FPTYPE* resultVariance);
} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
