/* file: kmeans_partialresult.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_PARTIALRESULT_
#define __KMEANS_PARTIALRESULT_

#include "algorithms/kmeans/kmeans_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{

/**
 * Allocates memory to store partial results of the K-Means algorithm
 * \param[in] input        Pointer to the structure of the input objects
 * \param[in] parameter    Pointer to the structure of the algorithm parameters
 * \param[in] method       Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT void PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);

    size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    size_t nClusters = kmPar->nClusters;

    Argument::set(nObservations, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(1, nClusters, data_management::NumericTable::doAllocate)));
    Argument::set(partialSums, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                data_management::NumericTable::doAllocate)));
    Argument::set(partialGoalFunction, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(1, 1, data_management::NumericTable::doAllocate)));

    if( kmPar->assignFlag )
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        size_t nRows = algInput->get(data)->getNumberOfRows();
        Argument::set(partialAssignments, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<int>(1, nRows, data_management::NumericTable::doAllocate)));
    }
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
