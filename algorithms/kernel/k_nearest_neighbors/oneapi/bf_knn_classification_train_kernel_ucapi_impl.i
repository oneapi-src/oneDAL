/* file: bf_knn_classification_train_kernel_ucapi_impl.i */
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

#ifndef __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_UCAPI_IMPL_I__
#define __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_UCAPI_IMPL_I__

#include "bf_knn_classification_train_kernel_ucapi.h"

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

template <typename algorithmFpType>
Status KNNClassificationTrainKernelUCAPI<algorithmFpType>::
                 compute(NumericTable * x, NumericTable * y, Model * r, const Parameter& par, engines::BatchBase &engine)
{
    return Status();
}

} // namespace internal
} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
