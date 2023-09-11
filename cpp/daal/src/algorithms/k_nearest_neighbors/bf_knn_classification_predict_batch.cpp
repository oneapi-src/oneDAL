/* file: bf_knn_classification_predict_batch.cpp */
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

#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"
#include "algorithms/classifier/classifier_model.h"
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
Input::Input() : classifier::prediction::Input() {}

bf_knn_classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<bf_knn_classification::Model, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void Input::set(classifier::prediction::ModelInputId id, const bf_knn_classification::ModelPtr & value)
{
    Argument::set(id, value);
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_STATUS_VAR(classifier::prediction::Input::check(parameter, method));

    const bf_knn_classification::ModelPtr m = get(classifier::prediction::model);
    ErrorCollection errors;
    errors.setCanThrow(false);
    DAAL_CHECK(checkNumericTable(m->impl()->getData().get(), dataStr()), ErrorModelNotFullInitialized);
    DAAL_CHECK_EX(algParameter->k <= m->impl()->getData()->getNumberOfRows(), services::ErrorIncorrectParameter, services::ParameterName, kStr());
    if ((algParameter->resultsToEvaluate & daal::algorithms::classifier::computeClassLabels) != 0)
    {
        DAAL_CHECK(checkNumericTable(m->impl()->getLabels().get(), labelsStr()), ErrorModelNotFullInitialized);
    }
    return services::Status();
}

} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
