/* file: gbt_regression_training_result.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_GBT_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::regression::training::Result(lastResultNumericTableId + 1) {};

gbt::regression::ModelPtr Result::get(ResultId id) const
{
    return gbt::regression::Model::cast(algorithms::regression::training::Result::get(algorithms::regression::training::ResultId(id)));
}

void Result::set(ResultId id, const gbt::regression::ModelPtr & value)
{
    algorithms::regression::training::Result::set(algorithms::regression::training::ResultId(id), value);
}

data_management::NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    return algorithms::regression::training::Result::check(input, par, method);
}

} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
