/* file: stump_classification_train_fpt.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "algorithms/stump/stump_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
namespace training
{
/**
 * Allocates memory to store final results of the decision stump training algorithm
 * \tparam algorithmFPType  Data type to store prediction results
 * \param[in] input         %Input objects for the decision stump training algorithm
 * \param[in] parameter     Parameters of the decision stump training algorithm
 * \param[in] method        Decision stump training method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::InputIface * algInput     = static_cast<const classifier::training::InputIface *>(input);
    const stump::classification::Parameter * algParameter = static_cast<const stump::classification::Parameter *>(parameter);
    services::Status st;
    stump::classification::ModelPtr modelPtr = stump::classification::Model::create(algInput->getNumberOfFeatures(), algParameter->nClasses, &st);
    DAAL_CHECK_STATUS_VAR(st);
    classifier::training::Result::set(classifier::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace training
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
