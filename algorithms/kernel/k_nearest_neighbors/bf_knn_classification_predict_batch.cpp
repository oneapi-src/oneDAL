/* file: bf_knn_classification_predict_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"
#include "oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "daal_strings.h"

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
namespace interface1
{

Input::Input() : classifier::prediction::Input() {}

bf_knn_classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<bf_knn_classification::interface1::Model, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void Input::set(classifier::prediction::ModelInputId id, const bf_knn_classification::interface1::ModelPtr & value)
{
    Argument::set(id, value);
}

services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_EX(algParameter->k > 0, services::ErrorIncorrectParameter, services::ParameterName, kStr());
    DAAL_CHECK_STATUS_VAR(classifier::prediction::Input::check(parameter, method));

    const bf_knn_classification::ModelPtr m = get(classifier::prediction::model);
    ErrorCollection errors;
    errors.setCanThrow(false);
    DAAL_CHECK(checkNumericTable(m->impl()->getData().get(), dataStr()), ErrorModelNotFullInitialized);
    DAAL_CHECK(checkNumericTable(m->impl()->getLabels().get(), labelsStr()), ErrorModelNotFullInitialized);
    return services::Status();
}

} // namespace interface1
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
