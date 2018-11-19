/* file: decision_tree_classification_training_result.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of Decision tree algorithm classes.
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_training_types.h"
#include "serialization_utils.h"

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
