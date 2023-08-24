/* file: elastic_net_training_types.cpp */
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
//  Implementation of elastic net algorithm classes.
//--
*/

#include "algorithms/elastic_net/elastic_net_training_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
Input::Input() : linear_model::training::Input(lastOptionalInputId + 1) {}
Input::Input(const Input & other) : linear_model::training::Input(other) {}

/**
 * Returns an input object for elastic net model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return linear_model::training::Input::get(linear_model::training::InputId(id));
}

/**
 * Sets an input object for elastic net model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & value)
{
    linear_model::training::Input::set(linear_model::training::InputId(id), value);
}

/**
* Returns optional input of the iterative solver algorithm
* \param[in] id    Identifier of the optional input data
* \return          %Input data that corresponds to the given identifier
*/
algorithms::OptionalArgumentPtr Input::get(OptionalInputId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets optional input for the iterative solver algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns input NumericTable containing optional data
* \param[in] id    Identifier of the input numeric table
* \return          %Input numeric table that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(elastic_net::training::optionalArgument);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

/**
* Sets optional input for the algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(OptionalDataId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(elastic_net::training::optionalArgument);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(elastic_net::training::lastOptionalData + 1));
        set(elastic_net::training::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNumberOfFeatures() const
{
    return get(data)->getNumberOfColumns();
}

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t Input::getNumberOfDependentVariables() const
{
    return get(dependentVariables)->getNumberOfColumns();
}

/**
* Checks an input object for the elastic net algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*
* \return Status of computations
*/
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Input::check(par, method));

    const NumericTablePtr dataTable = get(data);
    DAAL_CHECK(dataTable.get(), services::ErrorNullPtr);

    const NumericTablePtr dependentVariableTable = get(dependentVariables);
    const size_t nColumnsInDepVariable           = dependentVariableTable->getNumberOfColumns();

    const Parameter * parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    DAAL_CHECK_STATUS(s, parameter->check());

    const size_t penaltyL1NumberOfColumns = parameter->penaltyL1->getNumberOfColumns();
    DAAL_CHECK((penaltyL1NumberOfColumns == 1) || (nColumnsInDepVariable == penaltyL1NumberOfColumns), ErrorIncorrectNumberOfColumns);
    const size_t penaltyL2NumberOfColumns = parameter->penaltyL2->getNumberOfColumns();
    DAAL_CHECK((penaltyL2NumberOfColumns == 1) || (nColumnsInDepVariable == penaltyL2NumberOfColumns), ErrorIncorrectNumberOfColumns);
    return services::Status();
}

Parameter::Parameter(const SolverPtr & solver)
    : linear_model::Parameter(),
      penaltyL1(HomogenNumericTable<double>::create(1, 1, NumericTableIface::doAllocate, 0.5)),
      penaltyL2(HomogenNumericTable<double>::create(1, 1, NumericTableIface::doAllocate, 0.5)),
      optimizationSolver(solver),
      dataUseInComputation(doUse),
      optResultToCompute(0)
{}

services::Status Parameter::check() const
{
    services::Status status = checkNumericTable(penaltyL1.get(), penaltyL1Str(), packed_mask, 0, 0, 1);
    status                  = (status == services::Status() ? checkNumericTable(penaltyL2.get(), penaltyL2Str(), packed_mask, 0, 0, 1) : status);
    return status;
}

} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal
