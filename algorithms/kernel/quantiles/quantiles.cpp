/* file: quantiles.cpp */
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
//  Implementation of quantiles algorithm and types methods.
//--
*/

#include "quantiles_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace interface1
{
Parameter::Parameter(const data_management::NumericTablePtr quantileOrders)
    : daal::algorithms::Parameter(), quantileOrders(quantileOrders)
{
    if(quantileOrders.get() == NULL)
    {
        this->quantileOrders = data_management::NumericTablePtr(
                                   new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 0.5));
    }
}

Input::Input() : daal::algorithms::Input(1) {}

/**
 * Returns an input object for the quantiles algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the quantiles algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check the correctness of the %Input object
 * \param[in] parameter Pointer to the parameters structure
 * \param[in] method    Algorithm computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    if (!data_management::checkNumericTable(algParameter->quantileOrders.get(), this->_errors.get(),
        quantileOrdersStr(), 0, 0, 0, 1)) { return; }

    if (!data_management::checkNumericTable(get(data).get(), this->_errors.get(), dataStr())) { return; }
}

Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the final result of the quantiles algorithm
 * \param[in] id   Identifier of the final result, \ref ResultId
 * \return         Final result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the Result object of the quantiles algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the object
 * \param[in] par    Pointer to the parameters structure
 * \param[in] method Algorithm computation method
 */
void Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    const Input *input = static_cast<const Input *>(in);
    const Parameter *parameter = static_cast<const Parameter *>(par);

    if (!data_management::checkNumericTable(parameter->quantileOrders.get(), this->_errors.get(),
        quantileOrdersStr(), 0, 0, 0, 1)) { return; }

    size_t nVectors  = input->get(data)->getNumberOfColumns();
    size_t nFeatures = parameter->quantileOrders->getNumberOfColumns();

    int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                            (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                            (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

    if (!data_management::checkNumericTable(get(quantiles).get(), this->_errors.get(),
        quantilesStr(), unexpectedLayouts, 0, nFeatures, nVectors)) { return; }
}

}// namespace interface1
}// namespace quantiles
}// namespace algorithms
}// namespace daal
