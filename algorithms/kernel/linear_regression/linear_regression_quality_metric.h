/* file: linear_regression_quality_metric.h */
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
//  Implementation of the class defining the linear regression model
//--
*/

#ifndef __LIN_REG_QUALITY_METRIC_
#define __LIN_REG_QUALITY_METRIC_

#include "algorithms/linear_regression/linear_regression_single_beta_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{

/**
* Allocates memory to store
* \param[in] input   %Input object
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Algorithm method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const data_management::NumericTablePtr dependentVariableTable = (static_cast<const Input *>(input))->get(expectedResponses);
    const size_t nDepVariable = dependentVariableTable->getNumberOfColumns();

    services::Status st;
    set(rms,
        data_management::HomogenNumericTable<algorithmFPType>::create(nDepVariable, 1, data_management::NumericTableIface::doAllocate, 0, &st));
    DAAL_CHECK_STATUS_VAR(st);
    set(variance,
        data_management::HomogenNumericTable<algorithmFPType>::create(nDepVariable, 1, data_management::NumericTableIface::doAllocate, 0, &st));
    DAAL_CHECK_STATUS_VAR(st);

    const size_t nBeta = (static_cast<const Input *>(input))->get(model)->getBeta()->getNumberOfColumns();
    data_management::DataCollectionPtr coll(new data_management::DataCollection());

    for (size_t i = 0; i < nDepVariable; ++i)
    {
        coll->push_back(data_management::HomogenNumericTable<algorithmFPType>::create(nBeta, nBeta, data_management::NumericTableIface::doAllocate, 0, &st));
    }

    set(betaCovariances, coll);

    set(zScore,
        data_management::HomogenNumericTable<algorithmFPType>::create(nBeta, nDepVariable, data_management::NumericTableIface::doAllocate, 0, &st));
    DAAL_CHECK_STATUS_VAR(st);

    set(confidenceIntervals,
        data_management::HomogenNumericTable<algorithmFPType>::create(2*nBeta, nDepVariable, data_management::NumericTableIface::doAllocate, 0, &st));
    DAAL_CHECK_STATUS_VAR(st);

    set(inverseOfXtX,
        data_management::HomogenNumericTable<algorithmFPType>::create(nBeta, nBeta, data_management::NumericTableIface::doAllocate, 0, &st));

    return st;
}

} // namespace single_beta
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
