/* file: stump_regression_train_fpt.cpp */
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

#include "algorithms/stump/stump_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
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
    services::Status st;
    stump::regression::ModelPtr modelPtr = stump::regression::Model::create(&st);
    DAAL_CHECK_STATUS_VAR(st);
    daal::algorithms::regression::training::Result::set(daal::algorithms::regression::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace training
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
