/* file: kdtree_knn_classification_train_dense_default_batch_fpt_dispatcher_v2.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of K-Nearest Neighbors container.
//--
*/

#include "src/algorithms/k_nearest_neighbors/inner/kdtree_knn_classification_train_container_v2.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kdtree_knn_classification::training::interface2::BatchContainer, batch, DAAL_FPTYPE,
                                      kdtree_knn_classification::training::defaultDense)
} // namespace algorithms
} // namespace daal
