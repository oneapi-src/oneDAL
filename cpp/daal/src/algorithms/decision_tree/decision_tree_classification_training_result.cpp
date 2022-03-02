/* file: decision_tree_classification_training_result.cpp */
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
//  Implementation of Decision tree algorithm classes.
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_training_types.h"
#include "src/services/serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_TREE_CLASSIFICATION_TRAINING_RESULT_ID);

Result::Result() : classifier::training::Result() {}

/**
 * Returns the result of Decision tree model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::decision_tree::classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return decision_tree::classification::Model::cast(Argument::get(id));
}

} // namespace interface1
} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
