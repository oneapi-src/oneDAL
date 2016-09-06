/* file: objective_function_types.cpp */
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
//  Implementation of objective function classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/objective_function_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace interface1
{
Parameter::Parameter(const DAAL_UINT64 resultsToCompute) : resultsToCompute(resultsToCompute) {}

Parameter::Parameter(const Parameter &other) : resultsToCompute(other.resultsToCompute) {}

/** Default constructor */
Input::Input(size_t n) : daal::algorithms::Input(n)
{}

/**
 * Sets one input object for Objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Objective function
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::SharedPtr<services::Error> error(new services::Error());
    const Parameter *algParameter = static_cast<const Parameter *>(par);

    if(algParameter == 0)
    { this->_errors->add(services::ErrorNullParameterNotSupported); return; }

    error = checkTable(get(argument), "argument", 1);
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
}

/**
 * Checks the correctness of the numeric table
 * \param[in] nt              Pointer to the numeric table
 * \param[in] argumentName    Name of checked argument
 * \param[in] requiredRows    Number of required rows. If it equal 0 or not mentioned, the numeric table can't have 0 rows
 * \param[in] requiredColumns Number of required columns. If it equal 0 or not mentioned, the numeric table can't have 0 columns
 */
services::SharedPtr<services::Error> Input::checkTable(data_management::NumericTablePtr nt, const char *argumentName,
                                                size_t requiredRows, size_t requiredColumns) const
{
    services::SharedPtr<services::Error> error(new services::Error());
    if(!nt) { error->setId(services::ErrorNullInputNumericTable); }
    else if(nt->getNumberOfRows()    == 0) { error->setId(services::ErrorEmptyInputNumericTable); }
    else if(nt->getNumberOfColumns() == 0) { error->setId(services::ErrorEmptyInputNumericTable); }
    else if(requiredRows    != 0 && nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations);  }
    else if(requiredColumns != 0 && nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns);    }
    if(error->id() != services::NoErrorMessageFound) { error->addStringDetail(services::ArgumentName, argumentName);}
    return error;
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(1)
{}

/**
 * Sets the result of the Objective function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the numeric table with the result
 */
void Result::set(ResultId id, const data_management::DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the collection of the results of the Objective function
 * \param[in] id   Identifier of the result
 * \return         %Result that corresponds to the given identifier
 */
data_management::DataCollectionPtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the result numeric table from the Objective function recult collection by index
 * \param[in] id   Identifier of the result
 * \param[in] idx  Indentifier of index in result collection
 * \return         %Result that corresponds to the given identifiers
 */
data_management::NumericTablePtr Result::get(ResultId id, ResultCollectionId idx) const
{
    data_management::DataCollectionPtr collection = this->get(id);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*collection)[(int)idx]);
}

/**
* Checks the result of the Objective function
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method
*/
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    using namespace services;

    SharedPtr<Error> error(new Error());

    if(Argument::size() != 1) { this->_errors->add(ErrorIncorrectNumberOfArguments); return; }

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    if(algParameter == 0)
    { this->_errors->add(ErrorNullParameterNotSupported); return; }

    size_t nFeatures = algInput->get(argument)->getNumberOfColumns();

    if(algParameter->resultsToCompute & value)
    {
        error = checkTable(get(resultCollection, valueIdx), "value", 1, 1);
        if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
    }
    if(algParameter->resultsToCompute & gradient)
    {
        error = checkTable(get(resultCollection, gradientIdx), "gradient", 1, nFeatures);
        if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
    }
    if(algParameter->resultsToCompute & hessian)
    {
        error = checkTable(get(resultCollection, hessianIdx), "hessian", nFeatures, nFeatures);
        if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
    }
}

    /**
 * Checks the correctness of the numeric table
 * \param[in] nt              Pointer to the numeric table
 * \param[in] argumentName    Name of checked argument
 * \param[in] requiredRows    Number of required rows
 * \param[in] requiredColumns Number of required columns
 */
services::SharedPtr<services::Error> Result::checkTable(data_management::NumericTablePtr nt, const char *argumentName,
                                                size_t requiredRows, size_t requiredColumns) const
{
    services::SharedPtr<services::Error> error(new services::Error());
    if(!nt)                                              { error->setId(services::ErrorNullOutputNumericTable); }
    else if(nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations); }
    else if(nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns); }
    if(error->id() != services::NoErrorMessageFound)     { error->addStringDetail(services::ArgumentName, argumentName); }
    return error;
}

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
