/* file: kmeans_container.h */
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
//  Implementation of K-means algorithm container -- a class that contains
//  Lloyd K-means kernels for supported architectures.
//--
*/

#include "kmeans_types.h"
#include "kmeans_batch.h"
#include "kmeans_distributed.h"
#include "kmeans_lloyd_kernel.h"

#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansBatchKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input  *input  = static_cast<Input *>(_in );
    Result *result = static_cast<Result *>(_res);

    NumericTable *a[lastInputId + 1] =
    {
        input->get(data).get(),
        input->get(inputCentroids).get()
    };

    NumericTable *r[lastResultId + 1] =
    {
        result->get(centroids).get(),
        result->get(assignments).get(),
        result->get(objectiveFunction).get(),
        result->get(nIterations).get()
    };

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::KMeansBatchKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep1Kernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    Input         *input = static_cast<Input *>(_in  );
    PartialResult *pres  = static_cast<PartialResult *>(_pres);
    Parameter     *par   = static_cast<Parameter *>(_par );

    const size_t na = 2;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(input->get(data          ).get());
    a[1] = static_cast<NumericTable *>(input->get(inputCentroids).get());

    const size_t nr = 5 + (par->assignFlag != 0);
    NumericTable *r[6];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations      ).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums        ).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());
    if( par->assignFlag )
    {
        r[5] = static_cast<NumericTable *>(pres->get(partialAssignments).get());
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel,
                       __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult *pres  = static_cast<PartialResult *>(_pres);
    Result        *res   = static_cast<Result *>(_res );
    Parameter     *par   = static_cast<Parameter *>(_par );

    const size_t na = 1;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(pres->get(partialAssignments).get());

    const size_t nr = 1;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(res->get(assignments).get());

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel,
                       __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep2Kernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2MasterInput *input = static_cast<DistributedStep2MasterInput *>(_in);
    data_management::DataCollection *dcInput = input->get(partialResults).get();
    PartialResult *pres = static_cast<PartialResult *>(_pres);

    size_t nPartials = dcInput->size();

    size_t na = nPartials * 5;
    daal::internal::TArray<NumericTable *, cpu> aPtr(na);
    NumericTable **a = aPtr.get();
    DAAL_CHECK_MALLOC(a);
    for(size_t i = 0; i < nPartials; i++)
    {
        PartialResult *inPres = static_cast<PartialResult *>( (*dcInput)[i].get() );
        a[i * 5 + 0] = static_cast<NumericTable *>(inPres->get(nObservations).get());
        a[i * 5 + 1] = static_cast<NumericTable *>(inPres->get(partialSums    ).get());
        a[i * 5 + 2] = static_cast<NumericTable *>(inPres->get(partialObjectiveFunction    ).get());
        a[i * 5 + 3] = static_cast<NumericTable *>(inPres->get(partialCandidatesDistances  ).get());
        a[i * 5 + 4] = static_cast<NumericTable *>(inPres->get(partialCandidatesCentroids  ).get());
    }

    const size_t nr = 5;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums    ).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction  ).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par);

    dcInput->clear();
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult *pres = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    const size_t na = 5;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    a[1] = static_cast<NumericTable *>(pres->get(partialSums    ).get());
    a[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction    ).get());
    a[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances  ).get());
    a[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids  ).get());

    const size_t nr = 2;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());
    r[1] = static_cast<NumericTable *>(result->get(objectiveFunction).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep2Kernel,
                       __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r, par);
}

} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
