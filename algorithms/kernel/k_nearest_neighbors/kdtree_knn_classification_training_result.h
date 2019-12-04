/* file: kdtree_knn_classification_training_result.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAINING_RESULT_
#define __KDTREE_KNN_CLASSIFICATION_TRAINING_RESULT_

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
/**
 * Allocates memory to store the result of KD-tree based kNN model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of KD-tree based kNN model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, int method)
{
    services::Status status;
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, kdtree_knn_classification::ModelPtr(Model::create(algInput->getNumberOfFeatures(), &status)));
    return status;
}

/**
 * Allocates memory to store the result of KD-tree based kNN model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of KD-tree based kNN model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input,
                                              const kdtree_knn_classification::interface1::Parameter * parameter, int method)
{
    services::Status status;
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, kdtree_knn_classification::ModelPtr(Model::create(algInput->getNumberOfFeatures(), &status)));
    return status;
}

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
