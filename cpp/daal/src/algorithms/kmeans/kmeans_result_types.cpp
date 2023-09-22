/* file: kmeans_result_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_types.h"
#include "services/daal_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_KMEANS_RESULT_ID);
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the K-Means algorithm
 * \param[in] id   Result identifier
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the K-Means algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the K-Means algorithm
* \param[in] input   %Input objects for the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const Input * algInput = dynamic_cast<const Input *>(const_cast<daal::algorithms::Input *>(input));
    DAAL_CHECK(algInput, services::ErrorNullInput);

    size_t inputFeatures   = algInput->getNumberOfFeatures();
    const size_t inputRows = algInput->get(data)->getNumberOfRows();

    const interface2::Parameter * kmPar2 = dynamic_cast<const interface2::Parameter *>(par);
    if (kmPar2 == nullptr) return services::Status(daal::services::ErrorNullParameterNotSupported);
    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(objectiveFunction).get(), objectiveFunctionStr(), unexpectedLayouts, 0, 1, 1));

    DAAL_CHECK_STATUS(s, checkNumericTable(get(nIterations).get(), nIterationsStr(), unexpectedLayouts, 0, 1, 1));

    if (kmPar2)
    {
        if (kmPar2->resultsToEvaluate & computeCentroids)
        {
            DAAL_CHECK_STATUS(s, checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, inputFeatures, kmPar2->nClusters));
        }
        if (kmPar2->resultsToEvaluate & computeAssignments || kmPar2->assignFlag)
        {
            NumericTablePtr assignmentsTable = get(assignments);
            DAAL_CHECK_STATUS(s, checkNumericTable(assignmentsTable.get(), assignmentsStr(), unexpectedLayouts, 0, 1, inputRows));
        }
    }
    return s;
}

/**
 * Checks the results of the K-Means algorithm
 * \param[in] pres    Partial results of the algorithm
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * par, int method) const
{
    const interface2::Parameter * kmPar = static_cast<const interface2::Parameter *>(par);
    const int unexpectedLayouts         = (int)packed_mask;
    PartialResult * algPres             = static_cast<PartialResult *>(const_cast<daal::algorithms::PartialResult *>(pres));
    size_t presFeatures                 = algPres->get(partialSums)->getNumberOfColumns();
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, presFeatures, kmPar->nClusters));
    return checkNumericTable(get(objectiveFunction).get(), objectiveFunctionStr(), unexpectedLayouts, 0, 1, 1);
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal
