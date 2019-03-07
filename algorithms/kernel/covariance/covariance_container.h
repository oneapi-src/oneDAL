/* file: covariance_container.h */
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

/*
//++
//  Implementation of Covariance algorithm container.
//--
*/

#ifndef __COVARIANCE_CONTAINER_H__
#define __COVARIANCE_CONTAINER_H__

#include "kernel.h"
#include "covariance_batch.h"
#include "covariance_online.h"
#include "covariance_distributed.h"
#include "covariance_kernel.h"

#undef  __DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR
#define __DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(ComputeMethod, KernelClass)   \
    template<typename algorithmFPType, CpuType cpu>                                 \
    BatchContainer<algorithmFPType, ComputeMethod, cpu>::BatchContainer(            \
        daal::services::Environment::env *daalEnv)                                  \
    {                                                                               \
        __DAAL_INITIALIZE_KERNELS(KernelClass, algorithmFPType, ComputeMethod);      \
    }

#undef  __DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR
#define __DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(ComputeMethod)                 \
    template<typename algorithmFPType, CpuType cpu>                                 \
    BatchContainer<algorithmFPType, ComputeMethod, cpu>::~BatchContainer()          \
    {                                                                               \
        __DAAL_DEINITIALIZE_KERNELS();                                               \
    }

#undef  __DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE
#define __DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(ComputeMethod, KernelClass)       \
    template<typename algorithmFPType, CpuType cpu>                                 \
    services::Status BatchContainer<algorithmFPType, ComputeMethod, cpu>::compute() \
    {                                                                               \
        Result *result = static_cast<Result *>(_res);                               \
        Input *input = static_cast<Input *>(_in);                                   \
                                                                                    \
        NumericTable *dataTable = input->get(data).get();                           \
        NumericTable *covTable  = result->get(covariance).get();                    \
        NumericTable *meanTable = result->get(mean).get();                          \
                                                                                    \
        Parameter *parameter = static_cast<Parameter *>(_par);                      \
        daal::services::Environment::env &env = *_env;                              \
                                                                                    \
         __DAAL_CALL_KERNEL(env, KernelClass,                                       \
                    __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod),        \
                    compute, dataTable, covTable, meanTable, parameter);            \
    }


#undef  __DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(ComputeMethod, KernelClass)  \
    template<typename algorithmFPType, CpuType cpu>                                 \
    OnlineContainer<algorithmFPType, ComputeMethod, cpu>::OnlineContainer(          \
        daal::services::Environment::env *daalEnv)                                  \
    {                                                                               \
        __DAAL_INITIALIZE_KERNELS(KernelClass, algorithmFPType, ComputeMethod);     \
    }

#undef  __DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(ComputeMethod)                \
    template<typename algorithmFPType, CpuType cpu>                                 \
    OnlineContainer<algorithmFPType, ComputeMethod, cpu>::~OnlineContainer()        \
    {                                                                               \
        __DAAL_DEINITIALIZE_KERNELS();                                              \
    }

#undef  __DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(ComputeMethod, KernelClass)          \
    template<typename algorithmFPType, CpuType cpu>                                     \
    services::Status OnlineContainer<algorithmFPType, ComputeMethod, cpu>::compute()    \
    {                                                                                   \
        PartialResult *partialResult = static_cast<PartialResult *>(_pres);             \
        Input *input = static_cast<Input *>(_in);                                       \
                                                                                        \
        NumericTable *dataTable = input->get(data).get();                               \
                                                                                        \
        NumericTable *nObsTable         = partialResult->get(nObservations).get();      \
        NumericTable *crossProductTable = partialResult->get(crossProduct).get();       \
        NumericTable *sumTable          = partialResult->get(sum).get();                \
                                                                                        \
        Parameter *parameter = static_cast<Parameter *>(_par);                          \
        daal::services::Environment::env &env = *_env;                                  \
                                                                                        \
                                                                                        \
         __DAAL_CALL_KERNEL(env, KernelClass,                                           \
                   __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod),             \
                   compute, dataTable, nObsTable, crossProductTable, sumTable,          \
                   parameter);                                                          \
    }

#undef  __DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(ComputeMethod, KernelClass)       \
    template<typename algorithmFPType, CpuType cpu>                                          \
    services::Status OnlineContainer<algorithmFPType, ComputeMethod, cpu>::finalizeCompute() \
    {                                                                                        \
        PartialResult *partialResult = static_cast<PartialResult *>(_pres);                  \
        Result *result = static_cast<Result *>(_res);                                        \
                                                                                             \
        NumericTable *nObsTable         = partialResult->get(nObservations).get();           \
        NumericTable *crossProductTable = partialResult->get(crossProduct).get();            \
        NumericTable *sumTable          = partialResult->get(sum).get();                     \
                                                                                             \
        NumericTable *covTable  = result->get(covariance).get();                             \
        NumericTable *meanTable = result->get(mean).get();                                   \
                                                                                             \
        Parameter *parameter = static_cast<Parameter *>(_par);                               \
        daal::services::Environment::env &env = *_env;                                       \
                                                                                             \
         __DAAL_CALL_KERNEL(env, KernelClass,                                                \
                   __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod),                  \
                   finalizeCompute, nObsTable, crossProductTable,                            \
                   sumTable, covTable, meanTable, parameter);                                \
    }

namespace daal
{
namespace algorithms
{
namespace covariance
{

__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(defaultDense,    internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(singlePassDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(sumDense,        internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(fastCSR,         internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(singlePassCSR,   internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(sumCSR,          internal::CovarianceCSRBatchKernel)

__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(defaultDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(sumDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(fastCSR)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(sumCSR)

__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(defaultDense,    internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(singlePassDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(sumDense,        internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(fastCSR,         internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(singlePassCSR,   internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(sumCSR,          internal::CovarianceCSRBatchKernel)


__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(defaultDense,    internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(sumDense,        internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(fastCSR,         internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(singlePassCSR,   internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(sumCSR,          internal::CovarianceCSROnlineKernel)

__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(defaultDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(sumDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(fastCSR)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(sumCSR)

__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(defaultDense,    internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(sumDense,        internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(fastCSR,         internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(singlePassCSR,   internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(sumCSR,          internal::CovarianceCSROnlineKernel)

__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(defaultDense,    internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(sumDense,        internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(fastCSR,         internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(singlePassCSR,   internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(sumCSR,          internal::CovarianceCSROnlineKernel)

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::CovarianceDistributedKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    DistributedInput<step2Master> *input = static_cast<DistributedInput<step2Master> *>(_in);
    DataCollection *collection = input->get(partialResults).get();

    NumericTable *nObsTable         = partialResult->get(nObservations).get();
    NumericTable *crossProductTable = partialResult->get(crossProduct).get();
    NumericTable *sumTable          = partialResult->get(sum).get();

    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

     __DAAL_CALL_KERNEL(env, internal::CovarianceDistributedKernel,
                       __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, collection, nObsTable, crossProductTable, sumTable, parameter);

    collection->clear();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    NumericTable *nObsTable         = partialResult->get(nObservations).get();
    NumericTable *crossProductTable = partialResult->get(crossProduct).get();
    NumericTable *sumTable          = partialResult->get(sum).get();

    NumericTable *covTable  = result->get(covariance).get();
    NumericTable *meanTable = result->get(mean).get();

    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

     __DAAL_CALL_KERNEL(env, internal::CovarianceDistributedKernel,
                       __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       finalizeCompute, nObsTable, crossProductTable, sumTable, covTable, meanTable, parameter);
}

}
}
}

#endif
