/* file: logitboost_training_result.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the interface for LogitBoost model-based training
//--
*/

#ifndef __LOGITBOOST_TRAINING_RESULT_
#define __LOGITBOOST_TRAINING_RESULT_

#include "algorithms/boosting/logitboost_training_types.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
/**
 * Allocates memory to store final results of the LogitBoost training algorithm
 * \param[in] input         %Input of the LogitBoost training algorithm
 * \param[in] parameter     Parameters of the algorithm
 * \param[in] method        LogitBoost computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);

    services::Status s;
    ModelPtr model = Model::create(algInput->getNumberOfFeatures(), static_cast<const Parameter *>(parameter), &s);
    set(classifier::training::model, model);
    return s;
}
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
