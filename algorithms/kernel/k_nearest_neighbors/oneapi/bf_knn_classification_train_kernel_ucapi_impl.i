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

/*
//++
//  Implementation of auxiliary functions for K-Nearest Neighbors BF method.
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_UCAPI_IMPL_I__
#define __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_UCAPI_IMPL_I__

#include "daal_defines.h"
#include "threading.h"
#include "daal_atomic_int.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "service_rng.h"
#include "service_sort.h"
#include "numeric_table.h"
#include "bf_knn_classification_model_ucapi_impl.h"
#include "bf_knn_classification_train_kernel_ucapi.h"
#include "engine_batch_impl.h"

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

using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;


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
