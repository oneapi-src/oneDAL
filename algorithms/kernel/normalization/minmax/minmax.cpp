/* file: minmax.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "minmax_types.h"

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
namespace interface1
{

/** Default constructor */
Input::Input() : daal::algorithms::Input(1) {}

/**
 * Returns an input object for the min-max normalization algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the min-max normalization algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check the correctness of the %Input object
 * \param[in] par       Algorithm parameter
 * \param[in] method    Algorithm computation method
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    NumericTablePtr dataTable = get(data);
    if (!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    NumericTable::BasicStatisticsDataCollection &basicStatistics = dataTable->basicStatistics;
    NumericTablePtr minimumsTable = basicStatistics.get(NumericTableIface::minimum);
    NumericTablePtr maximumsTable = basicStatistics.get(NumericTableIface::maximum);

    size_t nColumns = dataTable->getNumberOfColumns();
    if (minimumsTable) {
        if (!checkNumericTable(minimumsTable.get(), this->_errors.get(),
                               basicStatisticsMinimumStr(), 0, 0, nColumns, 1)) { return; }
    }
    if (maximumsTable) {
        if (!checkNumericTable(maximumsTable.get(), this->_errors.get(),
                               basicStatisticsMaximumStr(), 0, 0, nColumns, 1)) { return; }
    }
}


Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the final result of the min-max normalization algorithm
 * \param[in] id   Identifier of the final result, daal::algorithms::normalization::minmax::ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the Result object of the min-max normalization algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the input object
 * \param[in] par    Pointer to the parameter object
 * \param[in] method Algorithm computation method
 */
void Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    const Input *input = static_cast<const Input *>(in);

    size_t nFeatures = input->get(data)->getNumberOfColumns();
    size_t nVectors  = input->get(data)->getNumberOfRows();

    int unexpectedLayouts = data_management::packed_mask;
    if (!data_management::checkNumericTable(get(normalizedData).get(), this->_errors.get(), normalizedDataStr(),
                                            unexpectedLayouts, 0, nFeatures, nVectors))
    {
        return;
    }
}

}// namespace interface1
}// namespace minmax
}// namespace normalization
}// namespace algorithms
}// namespace daal
