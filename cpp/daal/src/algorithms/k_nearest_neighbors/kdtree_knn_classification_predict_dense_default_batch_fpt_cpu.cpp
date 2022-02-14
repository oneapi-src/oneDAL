/* file: kdtree_knn_classification_predict_dense_default_batch_fpt_cpu.cpp */
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

/*
//++
//  Implementation of prediction stage of K-Nearest Neighbors algorithm.
//--
*/

#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch_impl.i"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface3
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace interface3

namespace internal
{
template class DAAL_EXPORT KNNClassificationPredictKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
