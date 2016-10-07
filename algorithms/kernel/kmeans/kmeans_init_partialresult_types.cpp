/* file: kmeans_init_partialresult_types.cpp */
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

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{

PartialResult::PartialResult() : daal::algorithms::PartialResult(2) {}


/**
 * Returns a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id   Identifier of the partial result
 * \return         Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the result table of the K-Means algorithm
* \return Number of features in the result table of the K-Means algorithm
*/
size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr clusters = get(partialClusters);
    return clusters->getNumberOfColumns();
}

/**
* Checks a partial result of computing initial clusters for the K-Means algorithm
* \param[in] input   %Input object for the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    const Parameter *kmPar = static_cast<const Parameter *>(par);

    int unexpectedLayouts = (int)packed_mask;

    if (!checkNumericTable(get(partialClusters).get(), this->_errors.get(), partialClustersStr(),
        unexpectedLayouts, 0, inputFeatures, kmPar->nClusters)) { return; }
    if (!checkNumericTable(get(partialClustersNumber).get(), this->_errors.get(), partialClustersNumberStr(),
        unexpectedLayouts, 0, 1, 1)) { return; }

    if(dynamic_cast<const Input*>(input))
    {
        DAAL_CHECK_EX(kmPar->nRowsTotal > 0, ErrorIncorrectParameter, ParameterName, nRowsTotalStr());
        DAAL_CHECK_EX(kmPar->nRowsTotal != kmPar->offset, ErrorIncorrectParameter, ParameterName, offsetStr());
    }
}

/**
 * Checks a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
void PartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    int unexpectedLayouts = (int)packed_mask;

    if (!checkNumericTable(get(partialClusters).get(), this->_errors.get(), partialClustersStr(),
        unexpectedLayouts, 0, 0, kmPar->nClusters)) { return; }
    if (!checkNumericTable(get(partialClustersNumber).get(), this->_errors.get(), partialClustersNumberStr(),
        unexpectedLayouts, 0, 1, 1)) { return; }
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
