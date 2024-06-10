/* file: bf_knn_classification_predict_fpt_ucapi.cpp */
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

#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
namespace internal
{
template class DAAL_EXPORT KNNClassificationPredictKernelUCAPI<DAAL_FPTYPE>;

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
