/* file: stump_train_fpt.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "stump_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory to store final results of the decision stump training algorithm
 * \tparam algorithmFPType  Data type to store prediction results
 * \param[in] input         %Input objects for the decision stump training algorithm
 * \param[in] parameter     Parameters of the decision stump training algorithm
 * \param[in] method        Decision stump training method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    algorithmFPType dummy = 1.0;
    classifier::training::Result::set(classifier::training::model, services::SharedPtr<weak_learner::Model>(new Model(dummy)));
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace training
}// namespace stump
}// namespace algorithms
}// namespace daal
