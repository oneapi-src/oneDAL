/* file: implicit_als_train_init_result.cpp */
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

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

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
namespace init
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_RESULT_ID);
Parameter::Parameter(size_t nFactors, size_t fullNUsers, size_t seed)
    : nFactors(nFactors), fullNUsers(fullNUsers), seed(seed), engine(engines::mt19937::Batch<>::create())
{}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nFactors > 0, ErrorIncorrectParameter, ParameterName, nFactorsStr());
    return services::Status();
}

/**
 * Returns the result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::implicit_als::ModelPtr Result::get(ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::implicit_als::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const daal::algorithms::implicit_als::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

} // namespace interface1
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
