/* file: implicit_als_train_result_fpt.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
/**
 * Allocates memory to store the results of the implicit ALS training algorithm
 * \param[in] input         Pointer to the input structure
 * \param[in] parameter     Pointer to the parameter structure
 * \param[in] method        Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const Input * algInput         = static_cast<const Input *>(input);
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);

    size_t nUsers = algInput->getNumberOfUsers();
    size_t nItems = algInput->getNumberOfItems();

    Status s;
    Argument::set(model, Model::create<algorithmFPType>(nUsers, nItems, *algParameter, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
