/* file: weak_learner_train.cpp */
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
//  Implementation of weak_learner algorithm and types methods.
//--
*/

#include "stump_training_types.h"
#include "serialization_utils.h"

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
void Result::set(classifier::training::ResultId id, daal::algorithms::weak_learner::ModelPtr &value)
{
    Argument::set(id, value);
}

}// namespace interface1
}// namespace training
}// namespace weak_learner
}// namespace algorithms
}// namespace daal
