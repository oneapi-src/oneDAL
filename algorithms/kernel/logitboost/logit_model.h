/* file: logit_model.h */
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
//  Implementation of class defining LogitBoost model.
//--
*/

#ifndef __LOGIT_MODEL_
#define __LOGIT_MODEL_

#include "algorithms/boosting/logitboost_model.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{

/**
 *  Constructs the logitBoost %Model
 * \tparam modelFPType  Data type to store logitBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(const Parameter *par, modelFPType dummy) : boosting::Model(), _nIterations(par->maxIterations) {}

} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
