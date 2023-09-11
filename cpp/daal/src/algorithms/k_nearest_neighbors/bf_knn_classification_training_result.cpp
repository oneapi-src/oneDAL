/* file: bf_knn_classification_training_result.cpp */
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

#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_K_NEAREST_NEIGHBOR_BF_TRAINING_RESULT_ID);

Result::Result() : classifier::training::Result() {}

daal::algorithms::bf_knn_classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::bf_knn_classification::Model, data_management::SerializationIface>(Argument::get(id));
}

} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
