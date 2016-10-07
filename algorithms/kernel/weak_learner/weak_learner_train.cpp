/* file: weak_learner_train.cpp */
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
//  Implementation of weak_learner algorithm and types methods.
//--
*/

#include "stump_training_types.h"

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
namespace interface1
{
Result::Result() {}

/**
 * Returns the model trained with the weak learner  algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the weak learner  algorithm
 */
services::SharedPtr<daal::algorithms::weak_learner::Model> Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::weak_learner::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the training stage of the weak learner algorithm
 * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
 * \param[in] value   Pointer to the training result
 */
void Result::set(classifier::training::ResultId id, services::SharedPtr<daal::algorithms::weak_learner::Model> &value)
{
    Argument::set(id, value);
}

}// namespace interface1
}// namespace training
}// namespace weak_learner
}// namespace algorithms
}// namespace daal
