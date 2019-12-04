/* file: bf_knn_classification_training_result.h */
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

#ifndef __BF_KNN_CLASSIFICATION_TRAINING_RESULT_
#define __BF_KNN_CLASSIFICATION_TRAINING_RESULT_

#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{

template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, int method)
{
    services::Status status;
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);
    bf_knn_classification::ModelPtr mptr(new Model(algInput->getNumberOfFeatures()));
    DAAL_CHECK(mptr, services::ErrorNullResult);
    set(classifier::training::model, mptr);
    return services::Status();
}

} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
