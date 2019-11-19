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

/*
//++
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

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

/** Default constructor */
Input::Input() : classifier::prediction::Input() {}

/**
 * Returns the input Model object in the prediction stage of the KD-tree based kNN algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
bf_knn_classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<bf_knn_classification::interface1::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the KD-tree based kNN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const bf_knn_classification::interface1::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_EX(algParameter->k > 0, services::ErrorIncorrectParameter, services::ParameterName, kStr());
    services::Status s = classifier::prediction::Input::check(parameter, method);
    if(!s) return s;

    const bf_knn_classification::ModelPtr m = get(classifier::prediction::model);
    ErrorCollection errors;
    errors.setCanThrow(false);
    s |= checkNumericTable(m->impl()->getData().get(), dataStr());
    DAAL_CHECK(s, ErrorModelNotFullInitialized);
    s |= checkNumericTable(m->impl()->getLabels().get(), labelsStr());
    DAAL_CHECK(s, ErrorModelNotFullInitialized);
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
