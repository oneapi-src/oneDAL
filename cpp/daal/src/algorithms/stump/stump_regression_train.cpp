/* file: stump_regression_train.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "algorithms/stump/stump_regression_training_types.h"
#include "algorithms/regression/regression_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_STUMP_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::regression::training::Result(lastResultId + 1) {}

daal::algorithms::stump::regression::ModelPtr Result::get(daal::algorithms::regression::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::stump::regression::Model, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(daal::algorithms::regression::training::ResultId id, daal::algorithms::stump::regression::ModelPtr & value)
{
    Argument::set(id, value);
}

data_management::NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const ModelConstPtr m = get(algorithms::regression::training::model);
    DAAL_CHECK(m, ErrorNullModel);

    const algorithms::regression::training::Input * algInput = static_cast<const algorithms::regression::training::Input *>(input);
    const NumericTablePtr dependentVariableTable             = algInput->get(algorithms::regression::training::dependentVariables);
    return checkNumericTable(dependentVariableTable.get(), dependentVariableStr(), 0, 0, 1);
}

} // namespace training
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
