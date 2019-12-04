/* file: kmeans_init_batch_container_v1.h */
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
//  Implementation of initializing the K-Means algorithm container.
//--
*/

#ifndef __KMEANS_INIT_CONTAINER_V1_H__
#define __KMEANS_INIT_CONTAINER_V1_H__

#include "kmeans/inner/kmeans_init_batch_v1.h"
#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const size_t na = 1;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());

    const size_t nr = 1;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    internal::Parameter internalPar;

    internalPar.nClusters          = par->nClusters;
    internalPar.seed               = par->seed;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds            = par->nRounds;
    internalPar.nTrials            = 1;

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansInitKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, internalPar,
                       *par->engine);
}

} // namespace interface1

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
