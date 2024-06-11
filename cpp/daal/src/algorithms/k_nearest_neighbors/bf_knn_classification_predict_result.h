/* file: bf_knn_classification_predict_result.h */
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_TRAINING_RESULT_
#define __BF_KNN_CLASSIFICATION_TRAINING_RESULT_

#include "algorithms/classifier/classifier_model.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
/**
 * Allocates memory for storing prediction results of brute-force kNN algorithm
 * \tparam  algorithmFPType     Data type for storing prediction results
 * \param[in] input     Pointer to the input objects of the classification algorithm
 * \param[in] parameter Pointer to the parameters of the classification algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method)
{
    services::Status s;

    const Parameter * const par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    const size_t nRows = (static_cast<const classifier::prediction::InputIface *>(input))->getNumberOfRows();

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (par->resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
    {
        if (deviceInfo.isCpu)
        {
            set(prediction,
                data_management::HomogenNumericTable<algorithmFPType>::create(1, nRows, data_management::NumericTableIface::doAllocate, &s));
        }
    }

    if (s.ok() && (par->resultsToCompute & computeIndicesOfNeighbors))
    {
        if (deviceInfo.isCpu)
        {
            set(indices, data_management::HomogenNumericTable<int>::create(par->k, nRows, data_management::NumericTableIface::doAllocate, &s));
        }
    }

    if (s.ok() && (par->resultsToCompute & computeDistances))
    {
        if (deviceInfo.isCpu)
        {
            set(distances,
                data_management::HomogenNumericTable<algorithmFPType>::create(par->k, nRows, data_management::NumericTableIface::doAllocate, &s));
        }
    }

    return s;
}
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
