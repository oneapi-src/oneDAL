/* file: kmeans_result_types.cpp */
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
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the K-Means algorithm
* \param[in] input   %Input objects for the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Input *algInput = dynamic_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t inputFeatures   = algInput->getNumberOfFeatures();

    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, inputFeatures, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(objectiveFunction).get(), goalFunctionStr(), unexpectedLayouts, 0, 1, 1));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nIterations).get(), nIterationsStr(), unexpectedLayouts, 0, 1, 1));
    if(kmPar->assignFlag)
    {
        NumericTablePtr assignmentsTable = get(assignments);
        const size_t inputRows = algInput->get(data)->getNumberOfRows();
        DAAL_CHECK_STATUS(s, checkNumericTable(get(assignments).get(), assignmentsStr(), unexpectedLayouts, 0, 1, inputRows));
    }
    return s;
}

/**
 * Checks the results of the K-Means algorithm
 * \param[in] pres    Partial results of the algorithm
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = (int)packed_mask;
    PartialResult *algPres = static_cast<PartialResult *>(const_cast<daal::algorithms::PartialResult *>(pres));
    size_t presFeatures = algPres->get(partialSums)->getNumberOfColumns();
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, presFeatures, kmPar->nClusters));
    return checkNumericTable(get(objectiveFunction).get(), goalFunctionStr(), unexpectedLayouts, 0, 1, 1);
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal
