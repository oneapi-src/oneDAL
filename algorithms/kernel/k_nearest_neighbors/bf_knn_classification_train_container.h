/* file: bf_knn_classification_train_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#ifndef __BF_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __BF_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_shared_ptr.h"
#include "bf_knn_classification_training_batch.h"
#include "oneapi/bf_knn_classification_train_kernel_ucapi.h"
#include "oneapi/bf_knn_classification_model_ucapi_impl.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{

using namespace daal::data_management;

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS_SYCL(internal::KNNClassificationTrainKernelUCAPI, DAAL_FPTYPE);
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    services::Status status;
    const classifier::training::Input * const input = static_cast<classifier::training::Input *>(_in);
    Result * const result = static_cast<Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);

    const bf_knn_classification::ModelPtr r = result->get(classifier::training::model);

    const bf_knn_classification::Parameter * const par = static_cast<bf_knn_classification::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    const bool copy = (par->dataUseInModel == doNotUse);
    status |= r->impl()->setData<algorithmFpType>(x, copy);
    status |= r->impl()->setLabels<algorithmFpType>(y, copy);
    DAAL_CHECK_STATUS_VAR(status);

    __DAAL_CALL_KERNEL_SYCL(env, internal::KNNClassificationTrainKernelUCAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFpType),    \
                       compute, r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), *par, *par->engine);
}

} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
