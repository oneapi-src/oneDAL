/* file: low_order_moments_partial_result.cpp */
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_MOMENTS_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Gets the number of columns in the partial result of the low order %moments algorithm
 * \return Number of columns in the partial result
 */
Status PartialResult::getNumberOfColumns(size_t & nCols) const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(partialMinimum));
    Status s              = checkNumericTable(ntPtr.get(), partialMinimumStr());
    nCols                 = (s ? ntPtr->getNumberOfColumns() : 0);
    return s;
}

/**
 * Returns the partial result of the low order %moments algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result of the low order %moments algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks correctness of the partial result
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    int unexpectedLayouts = (int)NumericTableIface::csrArray;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1));

    unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialMinimum).get(), partialMinimumStr(), unexpectedLayouts));

    size_t nFeatures = get(partialMinimum)->getNumberOfColumns();
    return checkImpl(nFeatures);
}

/**
 * Checks  the correctness of partial result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    size_t nFeatures = 0;
    DAAL_CHECK_STATUS(s, static_cast<const InputIface *>(input)->getNumberOfColumns(nFeatures));

    const int unexpectedLayouts = (int)NumericTableIface::csrArray;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1));
    return checkImpl(nFeatures);
}

services::Status PartialResult::checkImpl(size_t nFeatures) const
{
    services::Status s;
    const int unexpectedLayouts  = (int)packed_mask;
    const char * errorMessages[] = { partialMinimumStr(), partialMaximumStr(), partialSumStr(), partialSumSquaresStr(),
                                     partialSumSquaresCenteredStr() };

    for (size_t i = 1; i < lastPartialResultId + 1; i++)
        DAAL_CHECK_STATUS(s, checkNumericTable(get((PartialResultId)i).get(), errorMessages[i - 1], unexpectedLayouts, 0, nFeatures, 1));
    return s;
}

Parameter::Parameter(EstimatesToCompute _estimatesToCompute) : estimatesToCompute(_estimatesToCompute) {}

services::Status Parameter::check() const
{
    return Status();
}

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
