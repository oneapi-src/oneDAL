/* file: decision_tree_classification_predict_batch.cpp */
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
//  Implementation of the interface for Decision tree model-based prediction
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_predict_types.h"
#include "src/algorithms/decision_tree/decision_tree_classification_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace prediction
{
using namespace daal::data_management;
using namespace daal::services;

/** Default constructor */
Input::Input() : classifier::prediction::Input() {}

/**
 * Returns the input Model object in the prediction stage of the Decision tree algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return decision_tree::classification::Model::cast(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the Decision tree algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return classifier::prediction::Input::check(parameter, method);
}

} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
