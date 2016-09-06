/* file: kernel_function.cpp */
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
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "kernel_function_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{

ParameterBase::ParameterBase(size_t rowIndexX, size_t rowIndexY, size_t rowIndexResult, ComputationMode computationMode) :
    rowIndexX(rowIndexX), rowIndexY(rowIndexY), rowIndexResult(rowIndexResult), computationMode(computationMode) {}

Input::Input() : daal::algorithms::Input(2) {}

/**
* Returns the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the input object
*/
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

void Input::checkCSR() const
{
    int csrLayout = (int)data_management::NumericTableIface::csrArray;

    if (!data_management::checkNumericTable(get(X).get(), this->_errors.get(), XStr(), 0, csrLayout)) { return; }

    size_t nFeaturesX = get(X)->getNumberOfColumns();
    if (!data_management::checkNumericTable(get(Y).get(), this->_errors.get(), YStr(), 0, csrLayout, nFeaturesX)) { return; }
}

void Input::checkDense() const
{
    if (!data_management::checkNumericTable(get(X).get(), this->_errors.get(), XStr())) { return; }

    size_t nFeaturesX = get(X)->getNumberOfColumns();
    if (!data_management::checkNumericTable(get(Y).get(), this->_errors.get(), YStr(), 0, 0, nFeaturesX)) { return; }
}
/**
 * Returns the result of the kernel function algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the kernel function algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the kernel function algorithm
* \param[in] input   %Input objects of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
           int method) const
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    ParameterBase *algParameter = static_cast<ParameterBase *>(const_cast<daal::algorithms::Parameter *>(par));

    size_t nRowsX = algInput->get(X)->getNumberOfRows();
    size_t nRowsY = algInput->get(Y)->getNumberOfRows();

    int unexpectedLayouts = data_management::packed_mask;
    if (!data_management::checkNumericTable(get(values).get(), this->_errors.get(), valuesStr(), unexpectedLayouts, 0, 0, nRowsX)) { return; }

    size_t nVectorsValues = get(values)->getNumberOfRows();

    if(algParameter->rowIndexResult >= nVectorsValues)
    {
        services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectParameter));
        error->addStringDetail(services::ParameterName, rowIndexResultStr());
        this->_errors->add(error);
        return;
    }
    if(algParameter->rowIndexX >= nRowsX)
    {
        services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectParameter));
        error->addStringDetail(services::ParameterName, rowIndexXStr());
        this->_errors->add(error);
        return;
    }
    if(algParameter->rowIndexY >= nRowsY)
    {
        services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectParameter));
        error->addStringDetail(services::ParameterName, rowIndexYStr());
        this->_errors->add(error);
        return;
    }
}

}// namespace interface1
}// namespace kernel_function
}// namespace algorithms
}// namespace daal
