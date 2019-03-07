/* file: adaboost_training_batch.cpp */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#include "algorithms/boosting/adaboost_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ADABOOST_TRAINING_RESULT_ID);

/**
 * Returns the model trained with the AdaBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the AdaBoost algorithm
 */
daal::algorithms::adaboost::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::adaboost::Model, SerializationIface>(Argument::get(id));
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    Status s = classifier::training::Result::check(input, parameter, method);
    if(!s) return s;
    daal::algorithms::adaboost::ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
    return s;
}


} // namespace interface1
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal
