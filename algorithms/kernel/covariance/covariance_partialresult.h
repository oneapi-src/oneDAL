/* file: covariance_partialresult.h */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#ifndef __COVARIANCE_PARTIALRESULT_
#define __COVARIANCE_PARTIALRESULT_

#include "covariance_types.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{

/**
 * Allocates memory to store partial results of the correlation or variance-covariance matrix algorithm
 * \param[in] input     %Input objects of the algorithm
 * \param[in] parameter Parameters of the algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const InputIface *algInput = static_cast<const InputIface *>(input);
    size_t nColumns = algInput->getNumberOfFeatures();

    Argument::set(nObservations, data_management::NumericTablePtr(
                      new data_management::HomogenNumericTable<size_t>(1, 1, data_management::NumericTable::doAllocate)));
    Argument::set(crossProduct, data_management::NumericTablePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nColumns, nColumns, data_management::NumericTable::doAllocate)));
    Argument::set(sum, data_management::NumericTablePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nColumns, 1, data_management::NumericTable::doAllocate)));
}

} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
