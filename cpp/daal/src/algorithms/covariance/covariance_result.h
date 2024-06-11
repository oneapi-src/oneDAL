/* file: covariance_result.h */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#ifndef __COVARIANCE_RESULT_
#define __COVARIANCE_RESULT_

#include "algorithms/covariance/covariance_types.h"
#include "data_management/data/homogen_numeric_table.h"

using namespace daal::data_management;
namespace daal
{
namespace algorithms
{
namespace covariance
{
/**
 * Allocates memory to store final results of the correlation or variance-covariance matrix algorithm
 * \param[in] input     %Input objects of the algorithm
 * \param[in] parameter Parameters of the algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const Input * algInput = static_cast<const Input *>(input);
    size_t nColumns        = algInput->getNumberOfFeatures();
    services::Status status;

    set(covariance, HomogenNumericTable<algorithmFPType>::create(nColumns, nColumns, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);

    set(mean, HomogenNumericTable<algorithmFPType>::create(nColumns, 1, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

/**
 * Allocates memory for storing Covariance final results
 * \param[in] partialResult      Partial Results arguments of the covariance algorithm
 * \param[in] parameter          Parameters of the covariance algorithm
 * \param[in] method             Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                              const int method)
{
    const PartialResult * pres = static_cast<const PartialResult *>(partialResult);
    size_t nColumns            = pres->getNumberOfFeatures();
    services::Status status;

    set(covariance, HomogenNumericTable<algorithmFPType>::create(nColumns, nColumns, NumericTable::doAllocate, &status));
    set(mean, HomogenNumericTable<algorithmFPType>::create(nColumns, 1, NumericTable::doAllocate, &status));

    return status;
}

} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
