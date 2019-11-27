/* file: bf_knn_classification_predict_dense_default_batch_container.h */
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

#include "bf_knn_classification_predict.h"
#include "oneapi/bf_knn_classification_predict_kernel_ucapi.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS_SYCL(internal::KNNClassificationPredictKernelUCAPI, DAAL_FPTYPE);
}

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    const classifier::prediction::Input * const input = static_cast<const classifier::prediction::Input *>(_in);
    classifier::prediction::Result * const result = static_cast<classifier::prediction::Result *>(_res);

    const data_management::NumericTableConstPtr a = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr r = result->get(classifier::prediction::prediction);

    const daal::algorithms::Parameter * const par = _par;
    __DAAL_CALL_KERNEL_SYCL(env, internal::KNNClassificationPredictKernelUCAPI, \
                            __DAAL_KERNEL_ARGUMENTS(algorithmFpType),           \
                            compute, a.get(), m.get(), r.get(), par);
}

} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
