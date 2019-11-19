/* file: bf_knn_classification_train_fpt_ucapi.cpp */
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
//  Implementation of prediction stage of K-Nearest Neighbors algorithm.
//--
*/

#include "bf_knn_classification_predict_dense_default_batch.h"
#include "oneapi/bf_knn_classification_train_kernel_ucapi_impl.i"
#include "bf_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{

namespace internal
{

template class KNNClassificationTrainKernelUCAPI<DAAL_FPTYPE>;

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
