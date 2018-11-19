/* file: brownboost_training_batch.cpp */
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
//  Implementation of the interface for BrownBoost model-based training
//--
*/

#include "algorithms/boosting/brownboost_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_BROWNBOOST_TRAINING_RESULT_ID);

/**
 * Returns the model trained with the BrownBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the BrownBoost algorithm
 */
daal::algorithms::brownboost::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::brownboost::Model, SerializationIface>(Argument::get(id));
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s = classifier::training::Result::check(input, parameter, method);
    if(!s) return s;
    daal::algorithms::brownboost::ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
    return s;
}


} // namespace interface1
} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal
