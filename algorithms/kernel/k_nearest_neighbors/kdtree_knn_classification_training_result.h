/* file: kdtree_knn_classification_training_result.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, int method)
{
    services::Status status;
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, kdtree_knn_classification::ModelPtr(Model::create(algInput->getNumberOfFeatures(), &status)));
    return status;
}

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
