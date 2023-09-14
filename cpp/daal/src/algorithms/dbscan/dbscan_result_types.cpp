/* file: dbscan_result_types.cpp */
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
//  Implementation of DBSCAN classes.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "services/daal_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DBSCAN_RESULT_ID);

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the DBSCAN algorithm
 * \param[in] id   Result identifier
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the DBSCAN algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the DBSCAN algorithm
* \param[in] input   %Input objects for the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const Input * algInput   = dynamic_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    const Parameter * algPar = static_cast<const Parameter *>(par);

    if (!algInput)
    {
        return services::Status(services::ErrorIncorrectParameter);
    }

    const size_t nRows     = algInput->get(data)->getNumberOfRows();
    const size_t nFeatures = algInput->get(data)->getNumberOfColumns();

    const int unexpectedLayouts = (int)packed_mask;

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(assignments).get(), assignmentsStr(), unexpectedLayouts, 0, 1, nRows));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nClusters).get(), nClustersStr(), unexpectedLayouts, 0, 1, 1));

    if (algPar->resultsToCompute & computeCoreIndices)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(coreIndices).get(), coreIndicesStr(), unexpectedLayouts, 0, 1, 0, false));
    }

    if (algPar->resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(coreObservations).get(), coreObservationsStr(), unexpectedLayouts, 0, nFeatures, 0, false));
    }
    return s;
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal
