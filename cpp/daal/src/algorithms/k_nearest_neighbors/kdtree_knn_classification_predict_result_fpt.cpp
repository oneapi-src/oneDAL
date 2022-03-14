/* file: kdtree_knn_classification_predict_result_fpt.cpp */
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

#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_result.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, int method);
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
