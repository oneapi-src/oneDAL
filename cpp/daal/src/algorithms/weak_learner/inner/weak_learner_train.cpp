/* file: weak_learner_train.cpp */
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
//  Implementation of weak_learner algorithm and types methods.
//--
*/

#include "algorithms/weak_learner/weak_learner_training_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace weak_learner
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_WEAK_LEARNER_RESULT_ID);
Result::Result() {}

/**
 * Returns the model trained with the weak learner  algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the weak learner  algorithm
 */
daal::algorithms::weak_learner::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::weak_learner::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the training stage of the weak learner algorithm
 * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
 * \param[in] value   Pointer to the training result
 */
void Result::set(classifier::training::ResultId id, daal::algorithms::weak_learner::ModelPtr & value)
{
    Argument::set(id, value);
}

} // namespace training
} // namespace weak_learner
} // namespace algorithms
} // namespace daal
