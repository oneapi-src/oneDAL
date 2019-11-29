/* file: kdtree_knn_classification_train_container_v1.h */
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
//  Implementation of K-Nearest Neighbors container.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_V1_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_V1_H__

#include "kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_shared_ptr.h"
#include "kdtree_knn_classification_training_batch.h"
#include "kdtree_knn_classification_train_kernel.h"
#include "kdtree_knn_classification_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
using namespace daal::data_management;

namespace interface1
{
/**
 *  \brief Initialize list of K-Nearest Neighbors kernels with implementations for supported architectures
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainBatchKernel, algorithmFpType, method);
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate K-Nearest Neighbors model.
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    services::Status status;
    const classifier::training::Input * const input = static_cast<classifier::training::Input *>(_in);
    Result * const result                           = static_cast<Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);

    const kdtree_knn_classification::ModelPtr r = result->get(classifier::training::model);

    const kdtree_knn_classification::interface1::Parameter * const par = static_cast<kdtree_knn_classification::interface1::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    const bool copy = (par->dataUseInModel == doNotUse);
    status |= r->impl()->setData<algorithmFpType>(x, copy);
    status |= r->impl()->setLabels<algorithmFpType>(y, copy);
    if (status != services::Status())
    {
        return status;
    }

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute,
                       r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), *par->engine);
}
} // namespace interface1

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
