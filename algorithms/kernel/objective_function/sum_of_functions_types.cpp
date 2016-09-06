/* file: sum_of_functions_types.cpp */
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

#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{
namespace interface1
{
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    numberOfTerms(numberOfTerms),
    objective_function::Parameter(resultsToCompute),
    batchIndices(batchIndices) {}

Parameter::Parameter(const Parameter &other) :
    numberOfTerms(other.numberOfTerms),
    objective_function::Parameter(other.resultsToCompute),
    batchIndices(other.batchIndices) {}

/**
 * Checks the correctness of the parameter
 */
void Parameter::check() const
{
    if(batchIndices.get() != NULL)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(batchIndices->getNumberOfRows() != 1)    { error->setId(services::ErrorIncorrectNumberOfObservations); }
        if(error->id() != services::NoErrorMessageFound)
        {
            error->addStringDetail(services::ArgumentName, "batchIndices");
            this->_errors->add(error);
        }
        return;
    }

    if(numberOfTerms == 0)
    {
        this->_errors->add(services::ErrorZeroNumberOfTerms);
    }
}

/** Default constructor */
Input::Input(size_t n) : objective_function::Input(n)
{}

/**
 * Sets one input object for Sum of functions
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Sum of functions
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

} // namespace interface1
} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
