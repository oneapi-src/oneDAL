/* file: decision_tree_regression_training_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#include "algorithms/decision_tree/decision_tree_regression_training_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_TREE_REGRESSION_TRAINING_RESULT_ID);

Result::Result() : algorithms::regression::training::Result(lastResultId + 1) {}

/**
 * Returns the result of Decision tree model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
ModelPtr Result::get(ResultId id) const
{
    return staticPointerCast<decision_tree::regression::Model, SerializationIface>(Argument::get(id));
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
