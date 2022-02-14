/* file: dbscan_result.h */
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
//  Implementation of dbscan classes.
//--
*/

#ifndef __DBSCAN_RESULT__
#define __DBSCAN_RESULT__

#include "algorithms/dbscan/dbscan_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const Input * const algInput = static_cast<const Input *>(input);
    const Parameter * par        = static_cast<const Parameter *>(parameter);

    const size_t nRows     = algInput->get(data)->getNumberOfRows();
    const size_t nFeatures = algInput->get(data)->getNumberOfColumns();

    services::Status status;
    set(assignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    set(nClusters, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    if (par->resultsToCompute & computeCoreIndices)
    {
        set(coreIndices, HomogenNumericTable<int>::create(1, 0, NumericTable::notAllocate, &status));
    }

    if (par->resultsToCompute & computeCoreObservations)
    {
        set(coreObservations, HomogenNumericTable<algorithmFPType>::create(nFeatures, 0, NumericTable::notAllocate, &status));
    }

    return status;
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
