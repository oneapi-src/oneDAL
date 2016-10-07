/* file: adaboost_training_result.h */
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

#ifndef __ADABOOST_TRAINING_RESULT_
#define __ADABOOST_TRAINING_RESULT_

#include "algorithms/boosting/adaboost_training_types.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{

/**
 * Allocates memory to store final results of AdaBoost training
 * \param[in] input         %Input of the AdaBoost training algorithm
 * \param[in] parameter     Parameters of the algorithm
 * \param[in] method        AdaBoost computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    algorithmFPType dummy = 1.0;
    set(classifier::training::model, services::SharedPtr<adaboost::Model>(new Model(dummy)));
}

} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
