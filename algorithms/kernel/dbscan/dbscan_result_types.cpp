/* file: dbscan_result_types.cpp */
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
//  Implementation of DBSCAN classes.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "daal_defines.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the DBSCAN algorithm
* \param[in] input   %Input objects for the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *algInput = dynamic_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    const Parameter *algPar = static_cast<const Parameter *>(par);

    const size_t nRows = algInput->get(data)->getNumberOfRows();
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
} // namespace algorithm
} // namespace daal
