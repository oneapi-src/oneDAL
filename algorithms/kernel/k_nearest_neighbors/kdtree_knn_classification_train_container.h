/* file: kdtree_knn_classification_train_container.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of K-Nearest Neighbors container.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__

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

/**
 *  \brief Initialize list of K-Nearest Neighbors kernels with implementations for supported architectures
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
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
    const classifier::training::Input * const input = static_cast<classifier::training::Input *>(_in);
    Result * const result = static_cast<Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);

    const kdtree_knn_classification::ModelPtr r = result->get(classifier::training::model);

    const kdtree_knn_classification::Parameter * const par = static_cast<kdtree_knn_classification::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    const bool copy = (par->dataUseInModel == doNotUse);
    r->impl()->setData<algorithmFpType>(x, copy);
    r->impl()->setLabels<algorithmFpType>(y, copy);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),    \
                       compute, r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), *par, *par->engine);
}

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
