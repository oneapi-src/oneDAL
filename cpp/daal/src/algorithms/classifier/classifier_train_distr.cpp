/* file: classifier_train_distr.cpp */
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
//  Implementation of classifier training methods.
//--
*/

#include "algorithms/classifier/classifier_training_types.h"
#include "src/services/serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_CLASSIFIER_TRAINING_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {};

/**
 * Returns the partial result in the training stage of the classification algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return         Partial result that corresponds to the given identifier
 */
classifier::ModelPtr PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result in the training stage of the classification algorithm
 * \param[in] id    Identifier of the partial result, \ref PartialResultId
 * \param[in] value Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const daal::algorithms::classifier::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the PartialResult object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(input, parameter);
}

services::Status PartialResult::checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const
{
    daal::algorithms::classifier::ModelPtr m = get(partialModel);
    DAAL_CHECK(m, services::ErrorNullModel);
    return services::Status();
}

} // namespace training
} // namespace classifier
} // namespace algorithms
} // namespace daal
