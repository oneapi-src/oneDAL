/* file: logitboost_training_batch.cpp */
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
//  Implementation of the interface for LogitBoost model-based training
//--
*/

#include "algorithms/boosting/logitboost_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LOGITBOOST_TRAINING_RESULT_ID);
/**
 * Returns the model trained with the LogitBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the LogitBoost algorithm
 */
daal::algorithms::logitboost::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::logitboost::Model, data_management::SerializationIface>(Argument::get(id));
}


} // namespace interface1
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal
