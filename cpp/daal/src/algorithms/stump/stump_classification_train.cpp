/* file: stump_classification_train.cpp */
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
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_STUMP_CLASSIFICATION_TRAINING_RESULT_ID);
Result::Result() {}

/**
 * Returns the model trained with the Stump algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the Stump algorithm
 */
daal::algorithms::stump::classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::stump::classification::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the training stage of the stump algorithm
 * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
 * \param[in] value   Pointer to the training result
 */
void Result::set(classifier::training::ResultId id, daal::algorithms::stump::classification::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Check the correctness of the Result object
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameters structure
 * \param[in] method    Algorithm computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkImpl(input, parameter));
    const classifier::Parameter * algPar = static_cast<const classifier::Parameter *>(parameter);
    DAAL_CHECK(algPar->nClasses >= 2, services::ErrorModelNotFullInitialized);
    return services::Status();
}

data_management::NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

} // namespace training
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
