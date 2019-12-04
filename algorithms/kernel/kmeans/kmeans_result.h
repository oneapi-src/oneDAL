/* file: kmeans_result.h */
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

/*
//++
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_RESULT_
#define __KMEANS_RESULT_

#include "algorithms/kmeans/kmeans_types.h"
#include "execution_context.h"
#include "oneapi/internal/types.h"
#include "numeric_table_sycl_homogen.h"

using namespace daal::data_management;
using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
/**
 * Allocates memory to store the results of the K-Means algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    const Parameter * kmPar = static_cast<const Parameter *>(parameter);

    Input * algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->getNumberOfFeatures();
    size_t nClusters = kmPar->nClusters;

    services::Status status;

    if (deviceInfo.isCpu)
    {
        set(centroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
        set(objectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
        set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

        if (kmPar->assignFlag)
        {
            size_t nRows = algInput->get(data)->getNumberOfRows();
            set(assignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
        }
    }
    else
    {
        size_t nRows = algInput->get(data)->getNumberOfRows();

        // W/a
        set(centroids, SyclHomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
        set(objectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));

        set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
        set(assignments, SyclHomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    }

    return status;
}
/**
 * Allocates memory to store the results of the K-Means algorithm
 * \param[in] partialResult Pointer to the partial result structure
 * \param[in] parameter     Pointer to the structure of the algorithm parameters
 * \param[in] method        Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                              const int method)
{
    size_t nClusters = (static_cast<const Parameter *>(parameter))->nClusters;
    size_t nFeatures = (static_cast<const PartialResult *>(partialResult))->getNumberOfFeatures();

    services::Status status;
    set(centroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    set(objectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    return status;
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
