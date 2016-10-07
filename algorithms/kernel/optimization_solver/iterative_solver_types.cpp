/* file: iterative_solver_types.cpp */
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
//  Implementation of iterative solver classes.
//--
*/

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
namespace interface1
{

Parameter::Parameter(const sum_of_functions::BatchPtr &function_, const size_t nIterations_, const double accuracyThreshold_, bool optionalResultReq):
    nIterations(nIterations_), accuracyThreshold(accuracyThreshold_), optionalResultRequired(optionalResultReq)
{
    if(function_)
    {
        function = function_->clone();
    }
}

Parameter::Parameter(const Parameter &other) : nIterations(other.nIterations),
    accuracyThreshold(other.accuracyThreshold), optionalResultRequired(other.optionalResultRequired)
{
    if(other.function)
    {
        function = other.function->clone();
    }
}

void Parameter::check() const
{
    DAAL_CHECK_EX(function.get(), ErrorIncorrectParameter, ArgumentName, "function");
    DAAL_CHECK_EX(function->sumOfFunctionsParameter, ErrorNullParameterNotSupported, ArgumentName, "sumOfFunctionsParameter");
    DAAL_CHECK_EX(function->sumOfFunctionsInput, ErrorNullInput, ArgumentName, "sumOfFunctionsInput");
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorIncorrectParameter, ArgumentName, "accuracyThreshold");
}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

algorithms::OptionalArgumentPtr Input::get(OptionalInputId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(OptionalInputId id, const algorithms::OptionalArgumentPtr &ptr)
{
    Argument::set(id, ptr);
}

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(this->size() != 2) {this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }
    if(!checkNumericTable(get(inputArgument).get(), this->_errors.get(), inputArgumentStr(), 0, 0, 1)) {return;}
}

data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

algorithms::OptionalArgumentPtr Result::get(OptionalResultId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(OptionalResultId id, const algorithms::OptionalArgumentPtr &ptr)
{
    Argument::set(id, ptr);
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
                   int method) const
{
    if(Argument::size() != 3) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

    const Input *algInput = static_cast<const Input *>(input);
    size_t nRows = algInput->get(inputArgument)->getNumberOfRows();

    if(!checkNumericTable(get(minimum).get(), this->_errors.get(), minimumStr(), 0, 0, 1, nRows)) {return;}
    if(!checkNumericTable(get(nIterations).get(), this->_errors.get(), nIterationsStr(), 0, 0, 1, 1)) {return;}
}

} // namespace interface1
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
