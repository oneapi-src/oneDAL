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

Parameter::Parameter(const sum_of_functions::BatchPtr& function_, const size_t nIterations_, const double accuracyThreshold_, bool optionalResultReq):
    nIterations(nIterations_), accuracyThreshold(accuracyThreshold_), optionalResultRequired(optionalResultReq)
{
    if(function_)
        function = function_->clone();
}

Parameter::Parameter(const Parameter &other) : nIterations(other.nIterations),
    accuracyThreshold(other.accuracyThreshold), optionalResultRequired(other.optionalResultRequired)
{
    if(other.function)
        function = other.function->clone();
}

void Parameter::check() const
{
    if(!function.get())
    {
        this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, "function"));
        return;
    }

    if(!function->sumOfFunctionsParameter)
    {
        this->_errors->add(services::Error::create(services::ErrorNullParameterNotSupported, services::ArgumentName, "sumOfFunctionsParameter"));
        return;
    }

    if(!function->sumOfFunctionsInput)
    {
        this->_errors->add(services::Error::create(services::ErrorNullInput, services::ArgumentName, "sumOfFunctionsInput"));
        return;
    }

    if(accuracyThreshold < 0)
    {
        this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, "accuracyThreshold"));
        return;
    }
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

    services::ErrorPtr error = checkTable(get(inputArgument), "inputArgument", 1);
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
}

services::ErrorPtr Input::checkTable(const data_management::NumericTablePtr& nt, const char *argumentName,
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
    size_t nFeatures = algInput->get(inputArgument)->getNumberOfColumns();

    services::ErrorPtr error = checkTable(get(minimum), "minimum", 1, nFeatures);
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

    error = checkTable(get(nIterations), "nIterations", 1, 1);
    if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
}

services::ErrorPtr Result::checkTable(const data_management::NumericTablePtr& nt, const char *argumentName,
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
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
