/* file: adaboost_training_batch.cpp */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#include "algorithms/boosting/adaboost_training_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_MULTICLASS_ADABOOST_TRAINING_RESULT_ID);

Result::Result() : classifier::training::Result(lastResultNumericTableId + 1) {}

NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

daal::algorithms::adaboost::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::adaboost::Model, SerializationIface>(Argument::get(id));
}

void Result::set(classifier::training::ResultId id, const ModelPtr & value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s = classifier::training::Result::check(input, parameter, method);
    if (!s) return s;
    daal::algorithms::adaboost::ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
    return s;
}
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal
