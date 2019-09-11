/* file: decision_tree_classification_predict_batch.cpp */
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
//  Implementation of the interface for Decision tree model-based prediction
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_predict_types.h"
#include "decision_tree_classification_model_impl.h"

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
namespace interface1
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

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
