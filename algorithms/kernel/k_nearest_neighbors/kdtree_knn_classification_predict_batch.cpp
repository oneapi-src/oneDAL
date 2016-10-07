/* file: kdtree_knn_classification_predict_batch.cpp */
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
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"
#include "kdtree_knn_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
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
services::SharedPtr<kdtree_knn_classification::Model> Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<kdtree_knn_classification::interface1::Model, data_management::SerializationIface>(Argument::get(id));
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
void Input::set(classifier::prediction::ModelInputId id, const services::SharedPtr<kdtree_knn_classification::interface1::Model> & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    classifier::prediction::Input::check(parameter, method);
    if(this->_errors->size() != 0) { return; }

    SharedPtr<const kdtree_knn_classification::Model> m = get(classifier::prediction::model);
    ErrorCollection errors;
    errors.setCanThrow(false);
    DAAL_CHECK(checkNumericTable(m->impl()->getData().get(), &errors, dataStr()), ErrorModelNotFullInitialized);
    DAAL_CHECK(checkNumericTable(m->impl()->getLabels().get(), &errors, labelsStr()), ErrorModelNotFullInitialized);
    DAAL_CHECK(checkNumericTable(m->impl()->getKDTreeTable().get(), &errors, kdTreeTableStr(), 0, NumericTableIface::aos, 4),
               ErrorModelNotFullInitialized);
    const auto kdTreeNumberOfRows = m->impl()->getKDTreeTable()->getNumberOfRows();
    DAAL_CHECK(kdTreeNumberOfRows > 0, ErrorModelNotFullInitialized);
    DAAL_CHECK(m->impl()->getRootNodeIndex() < kdTreeNumberOfRows, ErrorModelNotFullInitialized);
}

} // namespace interface1
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
