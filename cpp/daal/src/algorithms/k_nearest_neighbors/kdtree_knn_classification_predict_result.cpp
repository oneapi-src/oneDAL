/* file: kdtree_knn_classification_predict_result.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of K-Nearest Neighbors (kNN) algorithm classes.
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"
#include "src/services/serialization_utils.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_K_NEAREST_NEIGHBOR_PREDICTION_RESULT_ID);

Result::Result() : classifier::prediction::Result(lastResultId + 1) {}

/**
 * Returns the result of KD-tree based kNN model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of KD-tree based kNN model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of prediction results of KD-tree based kNN algorithm
 * \param[in] input     Pointer to the the input object
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(input, parameter);
}

services::Status Result::checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const
{
    services::Status s;
    // const size_t nRows = (static_cast<const InputIface *>(input))->getNumberOfRows();

    // const Input * const in = static_cast<const Input *>(input);
    // classifier::ModelPtr m = in->get(model);
    // DAAL_CHECK(m, services::ErrorNullModel);

    // const Parameter * const par = static_cast<const Parameter *>(parameter);
    // DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    // if (par->resultsToEvaluate & computeClassLabels)
    //     DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(prediction).get(), predictionStr(), data_management::packed_mask, 0, 1, nRows));
    // if (par->resultsToEvaluate & computeClassProbabilities)
    //     DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(probabilities).get(), probabilitiesStr(), data_management::packed_mask, 0,
    //                                                             par->nClasses, nRows));
    // if (par->resultsToEvaluate & computeClassLogProbabilities)
    //     DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(logProbabilities).get(), logProbabilitiesStr(), data_management::packed_mask, 0,
    //                                                             par->nClasses, nRows));

    return s;
}
} // namespace interface1
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
