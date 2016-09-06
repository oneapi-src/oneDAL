/* file: kmeans_partialresult_types.cpp */
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

#include "algorithms/kmeans/kmeans_types.h"
#include "daal_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{

PartialResult::PartialResult() : daal::algorithms::PartialResult(4) {}

/**
 * Returns a partial result of the K-Means algorithm
 * \param[in] id   Identifier of the partial result
 * \return         Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the K-Means algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the Input data table
* \return Number of features in the Input data table
*/

size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr sums = get(partialSums);
    return sums->getNumberOfColumns();
}

/**
* Checks partial results of the K-Means algorithm
* \param[in] input   %Input object of the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{

    size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    int unexpectedLayouts = (int)packed_mask;

    if (!checkNumericTable(get(nObservations).get(), this->_errors.get(), nObservationsStr(), unexpectedLayouts, 0, 1, kmPar->nClusters)) { return; }
    if (!checkNumericTable(get(partialSums).get(), this->_errors.get(), partialSumsStr(), unexpectedLayouts, 0, inputFeatures, kmPar->nClusters)) { return; }
    if (!checkNumericTable(get(partialGoalFunction).get(), this->_errors.get(), partialGoalFunctionStr(), unexpectedLayouts, 0, 1, 1)) { return; }

    if( kmPar->assignFlag )
    {
        Input *algInput = dynamic_cast<Input*>(const_cast<daal::algorithms::Input *>(input));
        if( !algInput ) { return; }
        size_t nRows = algInput->get(data)->getNumberOfRows();
        if (!checkNumericTable(get(partialAssignments).get(), this->_errors.get(), partialAssignmentsStr(), unexpectedLayouts, 0, 1, nRows)) { return; }
    }
}

/**
 * Checks partial results of the K-Means algorithm
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
void PartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    int unexpectedLayouts = (int)packed_mask;

    if (!checkNumericTable(get(nObservations).get(), this->_errors.get(), nObservationsStr(), unexpectedLayouts, 0, 1, kmPar->nClusters)) { return; }
    if (!checkNumericTable(get(partialSums).get(), this->_errors.get(), partialSumsStr(), unexpectedLayouts, 0, 0, kmPar->nClusters)) { return; }
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal
