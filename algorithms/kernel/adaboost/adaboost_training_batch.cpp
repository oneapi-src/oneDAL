/* file: adaboost_training_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#include "algorithms/boosting/adaboost_training_types.h"

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

/**
 * Returns the model trained with the AdaBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the AdaBoost algorithm
 */
SharedPtr<daal::algorithms::adaboost::Model> Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::adaboost::Model, SerializationIface>(Argument::get(id));
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    classifier::training::Result::check(input, parameter, method);
    if(this->_errors->size() != 0) { return; }
    SharedPtr<daal::algorithms::adaboost::Model> m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
}


} // namespace interface1
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal
