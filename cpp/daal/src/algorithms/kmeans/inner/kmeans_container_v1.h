/* file: kmeans_container_v1.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __KMEANS_CONTAINER_V1_H__
#define __KMEANS_CONTAINER_V1_H__

#include "src/algorithms/kmeans/kmeans_lloyd_kernel.h"
#include "src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_batch_kernel_ucapi.h"
#include "src/algorithms/kmeans/inner/kmeans_batch_v1.h"
#include "src/algorithms/kmeans/inner/kmeans_distributed_v1.h"
#include "src/algorithms/kmeans/inner/kmeans_types_v1.h"
#include "services/internal/sycl/execution_context.h"
#include "src/data_management/service_numeric_table.h"

#include "oneapi/dal/network/network.hpp"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
DAAL_FORCEINLINE void convertParameter(interface1::Parameter & par1, interface2::Parameter & par2)
{
    par2.nClusters         = par1.nClusters;
    par2.maxIterations     = par1.maxIterations;
    par2.accuracyThreshold = par1.accuracyThreshold;
    par2.gamma             = par1.gamma;
    par2.distanceType      = par1.distanceType;
    par2.resultsToEvaluate = computeCentroids | computeExactObjectiveFunction;
    par2.assignFlag        = par1.assignFlag;
    if (par1.assignFlag)
    {
        par2.resultsToEvaluate |= computeAssignments;
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::KMeansBatchKernel, method, algorithmFPType);
    }
    else
    {
        _kernel = new internal::KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>();
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * a[lastInputId + 1] = { input->get(data).get(), input->get(inputCentroids).get() };

    NumericTable * r[lastResultId + 1] = { result->get(centroids).get(), result->get(assignments).get(), result->get(objectiveFunction).get(),
                                           result->get(nIterations).get() };

    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter par2(par->nClusters, par->maxIterations);
    convertParameter(*par, par2);
    daal::services::Environment::env & env = *_env;

    oneapi::dal::network::empty_network net;

    if (deviceInfo.isCpu || method != lloydDense)
    {
        __DAAL_CALL_KERNEL(env, internal::KMeansBatchKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, r, &par2, net);
    }
    else
    {
        return ((internal::KMeansDenseLloydBatchKernelUCAPI<algorithmFPType> *)(_kernel))->compute(a, r, &par2);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep1Kernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    Input * input               = static_cast<Input *>(_in);
    PartialResult * pres        = static_cast<PartialResult *>(_pres);
    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter par2(par->nClusters, par->maxIterations);
    convertParameter(*par, par2);

    const size_t na = 2;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());
    a[1] = static_cast<NumericTable *>(input->get(inputCentroids).get());

    const size_t nr = 5 + (par->assignFlag != 0);
    NumericTable * r[6];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());
    if (par->assignFlag)
    {
        r[5] = static_cast<NumericTable *>(pres->get(partialAssignments).get());
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, &par2);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * pres = static_cast<PartialResult *>(_pres);
    Result * res         = static_cast<Result *>(_res);

    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter par2(par->nClusters, par->maxIterations);
    convertParameter(*par, par2);

    const size_t na = 1;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(pres->get(partialAssignments).get());

    const size_t nr = 1;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(res->get(assignments).get());

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r,
                       &par2);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep2Kernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2MasterInput * input       = static_cast<DistributedStep2MasterInput *>(_in);
    data_management::DataCollection * dcInput = input->get(partialResults).get();
    PartialResult * pres                      = static_cast<PartialResult *>(_pres);

    size_t nPartials = dcInput->size();

    size_t na = nPartials * 5;
    daal::internal::TArray<NumericTable *, cpu> aPtr(na);
    NumericTable ** a = aPtr.get();
    DAAL_CHECK_MALLOC(a);
    for (size_t i = 0; i < nPartials; i++)
    {
        PartialResult * inPres = static_cast<PartialResult *>((*dcInput)[i].get());
        a[i * 5 + 0]           = static_cast<NumericTable *>(inPres->get(nObservations).get());
        a[i * 5 + 1]           = static_cast<NumericTable *>(inPres->get(partialSums).get());
        a[i * 5 + 2]           = static_cast<NumericTable *>(inPres->get(partialObjectiveFunction).get());
        a[i * 5 + 3]           = static_cast<NumericTable *>(inPres->get(partialCandidatesDistances).get());
        a[i * 5 + 4]           = static_cast<NumericTable *>(inPres->get(partialCandidatesCentroids).get());
    }

    const size_t nr = 5;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());

    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter par2(par->nClusters, par->maxIterations);
    convertParameter(*par, par2);
    daal::services::Environment::env & env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
                                                   compute, na, a, nr, r, &par2);

    dcInput->clear();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * pres = static_cast<PartialResult *>(_pres);
    Result * result      = static_cast<Result *>(_res);

    const size_t na = 5;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    a[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    a[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    a[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    a[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());

    const size_t nr = 2;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());
    r[1] = static_cast<NumericTable *>(result->get(objectiveFunction).get());

    interface1::Parameter * par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter par2(par->nClusters, par->maxIterations);
    convertParameter(*par, par2);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r,
                       &par2);
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
