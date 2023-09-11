/* file: brownboost_training_batch.cpp */
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
//  Implementation of the interface for BrownBoost model-based training
//--
*/

#include "algorithms/boosting/brownboost_training_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_BROWNBOOST_TRAINING_RESULT_ID);

/**
 * Returns the model trained with the BrownBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the BrownBoost algorithm
 */
daal::algorithms::brownboost::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::brownboost::Model, SerializationIface>(Argument::get(id));
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s = classifier::training::Result::check(input, parameter, method);
    if (!s) return s;
    daal::algorithms::brownboost::ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
    return s;
}

} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal
