/* file: decision_tree_classification_training_result.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAINING_RESULT_
#define __DECISION_TREE_CLASSIFICATION_TRAINING_RESULT_

#include "algorithms/decision_tree/decision_tree_classification_training_types.h"

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
/**
 * Allocates memory to store the result of Decision tree model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of Decision tree model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input,
                                              const decision_tree::classification::interface1::Parameter * parameter, int method)
{
    services::Status status;
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, ModelPtr((Model::create(algInput->getNumberOfFeatures(), &status))));
    return status;
}

/**
 * Allocates memory to store the result of Decision tree model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of Decision tree model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const decision_tree::classification::Parameter * parameter,
                                              int method)
{
    services::Status status;
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, ModelPtr((Model::create(algInput->getNumberOfFeatures(), &status))));
    return status;
}

} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
