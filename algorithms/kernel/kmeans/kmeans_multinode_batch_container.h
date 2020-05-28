/* file: kmeans_multinode_batch_container.h */
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
//  Implementation of K-means algorithm container -- a class that contains
//  Lloyd K-means kernels for supported architectures.
//--
*/

#ifndef __KMEANS_CONTAINER_H__
#define __KMEANS_CONTAINER_H__

#include "algorithms/kmeans/kmeans_types.h"
#include "algorithms/kmeans/kmeans_multinode_batch.h"
#include "algorithms/kernel/kmeans/oneapi/kmeans_oneccl_dense_lloyd_batch_kernel_ucapi.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace preview
{
namespace kmeans
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
MultiNodeBatchContainer<algorithmFPType, method, cpu>::MultiNodeBatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS_SYCL(internal::KMeansOneCclDefaultBatchKernelUCAPI, DAAL_FPTYPE);
}

template <typename algorithmFpType, Method method, CpuType cpu>
MultiNodeBatchContainer<algorithmFpType, method, cpu>::~MultiNodeBatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
daal::services::Status MultiNodeBatchContainer<algorithmFpType, method, cpu>::compute()
{
    auto & context    = daal::services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    algorithms::kmeans::Input * const input   = static_cast<algorithms::kmeans::Input *>(_in);
    algorithms::kmeans::Result * const result = static_cast<algorithms::kmeans::Result *>(_res);

    NumericTable * a[algorithms::kmeans::lastInputId + 1] = { input->get(algorithms::kmeans::data).get(),
                                                              input->get(algorithms::kmeans::inputCentroids).get() };

    NumericTable * r[algorithms::kmeans::lastResultId + 1] = { result->get(algorithms::kmeans::centroids).get(),
                                                               result->get(algorithms::kmeans::assignments).get(),
                                                               result->get(algorithms::kmeans::objectiveFunction).get(),
                                                               result->get(algorithms::kmeans::nIterations).get() };

    algorithms::Parameter * par            = static_cast<algorithms::Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    return ((internal::KMeansOneCclDefaultBatchKernelUCAPI<algorithmFpType> *)(_kernel))->compute(a, r, par);
}

} // namespace interface1
} // namespace kmeans
} // namespace preview
} // namespace algorithms
} // namespace daal

#endif
