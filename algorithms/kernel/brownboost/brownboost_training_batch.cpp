/* file: brownboost_training_batch.cpp */
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
//  Implementation of the interface for BrownBoost model-based training
//--
*/

#include "algorithms/boosting/brownboost_training_types.h"

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

/**
 * Returns the model trained with the BrownBoost algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the BrownBoost algorithm
 */
SharedPtr<daal::algorithms::brownboost::Model> Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::brownboost::Model, SerializationIface>(Argument::get(id));
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    classifier::training::Result::check(input, parameter, method);
    if(this->_errors->size() != 0) { return; }
    SharedPtr<daal::algorithms::brownboost::Model> m = get(classifier::training::model);
    DAAL_CHECK(m->getAlpha(), ErrorModelNotFullInitialized);
}


} // namespace interface1
} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal
