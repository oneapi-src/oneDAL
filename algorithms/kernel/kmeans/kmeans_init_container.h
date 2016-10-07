/* file: kmeans_init_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "kmeans_init_types.h"
#include "kmeans_init_batch.h"
#include "kmeans_init_distributed.h"
#include "kmeans_init_kernel.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansinitKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    const size_t na = 1;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());

    const size_t nr = 1;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansinitKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansinitStep1LocalKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    Input *input        = static_cast<Input *>(_in  );
    PartialResult *pres = static_cast<PartialResult *>(_pres);

    const size_t na = 1;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(input->get(data            ).get());

    const size_t nr = 2;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(pres->get(partialClustersNumber).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialClusters      ).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansinitStep1LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute() {}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansinitStep2MasterKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2MasterInput *input = static_cast<DistributedStep2MasterInput *>(_in);
    data_management::DataCollection *dcInput = input->get(partialResults).get();
    PartialResult *pres = static_cast<PartialResult *>(_pres);

    size_t nPartials = dcInput->size();

    size_t na = nPartials * 2;
    NumericTable **a = new NumericTable*[na];
    for(size_t i = 0; i < nPartials; i++)
    {
        PartialResult *inPres = static_cast<PartialResult *>( (*dcInput)[i].get() );
        a[i * 2 + 0] = static_cast<NumericTable *>(inPres->get(partialClustersNumber).get());
        a[i * 2 + 1] = static_cast<NumericTable *>(inPres->get(partialClusters      ).get());
    }

    const size_t nr = 2;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(pres->get(partialClustersNumber).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialClusters      ).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansinitStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r,
                       par);

    delete[] a;

    dcInput->clear();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult *pres = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    const size_t na = 2;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(pres->get(partialClustersNumber).get());
    a[1] = static_cast<NumericTable *>(pres->get(partialClusters      ).get());

    const size_t nr = 1;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansinitStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr,
                       r, par);
}

} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
