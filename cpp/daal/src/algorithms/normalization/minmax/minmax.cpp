/* file: minmax.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "algorithms/normalization/minmax_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NORMALIZATION_MINMAX_RESULT_ID);

/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}

/**
 * Returns an input object for the min-max normalization algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets the input object of the min-max normalization algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check the correctness of the %Input object
 * \param[in] par       Algorithm parameter
 * \param[in] method    Algorithm computation method
 */
Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    NumericTablePtr dataTable = get(data);
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    NumericTable::BasicStatisticsDataCollection & basicStatistics = dataTable->basicStatistics;
    NumericTablePtr minimumsTable                                 = basicStatistics.get(NumericTableIface::minimum);
    NumericTablePtr maximumsTable                                 = basicStatistics.get(NumericTableIface::maximum);

    size_t nColumns = dataTable->getNumberOfColumns();
    if (minimumsTable)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(minimumsTable.get(), basicStatisticsMinimumStr(), 0, 0, nColumns, 1));
    }
    if (maximumsTable)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(maximumsTable.get(), basicStatisticsMaximumStr(), 0, 0, nColumns, 1));
    }
    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the final result of the min-max normalization algorithm
 * \param[in] id   Identifier of the final result, daal::algorithms::normalization::minmax::ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets the Result object of the min-max normalization algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the input object
 * \param[in] par    Pointer to the parameter object
 * \param[in] method Algorithm computation method
 */
Status Result::check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const
{
    const Input * input = static_cast<const Input *>(in);
    DAAL_CHECK(input, ErrorNullInput);

    const size_t nFeatures = input->get(data)->getNumberOfColumns();
    const size_t nVectors  = input->get(data)->getNumberOfRows();

    const int unexpectedLayouts = packed_mask;
    return checkNumericTable(get(normalizedData).get(), normalizedDataStr(), unexpectedLayouts, 0, nFeatures, nVectors);
}

} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
