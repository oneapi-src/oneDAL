/* file: adaboost_training_result_v1.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __ADABOOST_TRAINING_RESULT_V1_H__
#define __ADABOOST_TRAINING_RESULT_V1_H__

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
 * Allocates memory to store final results of AdaBoost training
 * \param[in] input         %Input of the AdaBoost training algorithm
 * \param[in] parameter     Parameters of the algorithm
 * \param[in] method        AdaBoost computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    const classifier::training::interface1::Input * algInput = static_cast<const classifier::training::interface1::Input *>(input);
    set(classifier::training::model, daal::algorithms::adaboost::interface1::Model::create<algorithmFPType>(algInput->getNumberOfFeatures(), &s));
    return s;
}
} // namespace interface1
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
