/* file: pca_dense_correlation_distr_step2_container.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_DISTR_STEP2_CONTAINER_H__
#define __PCA_DENSE_CORRELATION_DISTR_STEP2_CONTAINER_H__

#include "kernel.h"
#include "pca_distributed.h"
#include "pca_dense_correlation_distr_step2_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::DistributedContainer(daal::services::Environment::env
                                                                                                *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCACorrelationKernel, distributed, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::compute()
{
    DistributedInput<correlationDense> *input = static_cast<DistributedInput<correlationDense> *>(_in);
    PartialResult<correlationDense> *partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    DistributedParameter<step2Master, algorithmFPType, correlationDense> *parameter =
        static_cast<DistributedParameter<step2Master, algorithmFPType, correlationDense> *>(_par);
    services::Environment::env &env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(distributed, algorithmFPType),
                       compute, input, partialResult, parameter);

    input->get(partialResults)->clear();
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::finalizeCompute()
{
    PartialResult<correlationDense> *partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    Result *result = static_cast<Result *>(_res);
    DistributedParameter<step2Master, algorithmFPType, correlationDense> *parameter =
        static_cast<DistributedParameter<step2Master, algorithmFPType, correlationDense> *>(_par);

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(distributed, algorithmFPType),
                       finalize, partialResult, parameter, *eigenvectors, *eigenvalues);
}

}
}
} // namespace daal

#endif
