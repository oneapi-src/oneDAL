/* file: decision_tree_model.h */
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
//  Implementation of the class defining the Decision tree
//--
*/

#ifndef __DECISION_TREE_MODEL_H__
#define __DECISION_TREE_MODEL_H__

namespace daal
{
namespace algorithms
{
/**
 * @defgroup decision_tree Base Decision Tree
 * \brief Contains base classes for Decision tree algorithm
 * @ingroup training_and_prediction
 * @{
 */

/**
 * \brief Contains classes for Decision tree algorithm
 */
namespace decision_tree
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PRUNING"></a>
 * \brief Pruning method for Decision tree algorithm
 */
enum Pruning
{
    none                = 0, /*!< Do not prune */
    reducedErrorPruning = 1  /*!< Reduced error pruning */
};

} // namespace decision_tree

/** @} */
} // namespace algorithms
} // namespace daal

#endif
