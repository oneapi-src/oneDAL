/* file: mse_types.cpp */
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
//  Implementation of mse classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/mse_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace interface1
{
/**
 * Constructs the parameter of Mean squared error objective function
 * \param[in] numberOfTerms    The number of terms in the function
 * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                               a batch of indices used to compute the function results, e.g.,
                               value of the sum of the functions. If no indices are provided,
                               all terms will be used in the computations.
 * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
 */
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices, const DAAL_UINT64 resultsToCompute) :
                     sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute)
{}

/**
 * Constructs an Parameter by copying input objects and parameters of another Parameter
 * \param[in] other An object to be used as the source to initialize object
 */
Parameter::Parameter(const Parameter &other) :
    sum_of_functions::Parameter(other)
{}

/**
 * Checks the correctness of the parameter
 */
void Parameter::check() const
{
    sum_of_functions::Parameter::check();
}

/** Default constructor */
Input::Input() : sum_of_functions::Input(3)
{}

/**
 * Sets one input object for Mean squared error objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Mean squared error objective function
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
    sum_of_functions::Input::check(par, method);
    if(Argument::size() != 3)
    { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    services::SharedPtr<services::Error> error(new services::Error());

    error = checkTable(get(data), "data");
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

    size_t nRowsInData = get(data)->getNumberOfRows();

    error = checkTable(get(dependentVariables), "dependentVariables", nRowsInData, 1);
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

    error = checkTable(get(argument), "argument", 0, get(data)->getNumberOfColumns() + 1);
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
} // namespace mse
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
