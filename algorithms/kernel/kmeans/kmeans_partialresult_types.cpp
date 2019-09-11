/* file: kmeans_partialresult_types.cpp */
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

/*
//++
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_types.h"
#include "daal_defines.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_KMEANS_PARTIAL_RESULT_ID);
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

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
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = (int)packed_mask;

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, inputFeatures, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialGoalFunction).get(), partialGoalFunctionStr(), unexpectedLayouts, 0, 1, 1));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialCandidatesDistances).get(), partialCandidatesDistancesStr(), unexpectedLayouts, 0, 1, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialCandidatesCentroids).get(), partialCandidatesCentroidsStr(), unexpectedLayouts, 0, inputFeatures, kmPar->nClusters));
    if( kmPar->assignFlag )
    {
        Input *algInput = dynamic_cast<Input*>(const_cast<daal::algorithms::Input *>(input));
        if( !algInput ) { return s; }
        const size_t nRows = algInput->get(data)->getNumberOfRows();
        s = checkNumericTable(get(partialAssignments).get(), partialAssignmentsStr(), unexpectedLayouts, 0, 1, nRows);
    }
    return s;
}

/**
 * Checks partial results of the K-Means algorithm
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, 0, kmPar->nClusters));
    return s;
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal
