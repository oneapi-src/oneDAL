/* file: kdtree_knn_classification_training_result.cpp */
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
//  Implementation of K-Nearest Neighbors (kNN) algorithm classes.
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_K_NEAREST_NEIGHBOR_TRAINING_RESULT_ID);

Result::Result() : classifier::training::Result() {}

/**
 * Returns the result of KD-tree based kNN model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::kdtree_knn_classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::kdtree_knn_classification::Model,
                                      data_management::SerializationIface>(Argument::get(id));
}

} // namespace interface1
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
