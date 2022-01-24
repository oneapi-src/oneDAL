/* file: pca_dense_correlation_distr_step2_container.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_DISTR_STEP2_CONTAINER_H__
#define __PCA_DENSE_CORRELATION_DISTR_STEP2_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_distributed.h"
#include "src/algorithms/pca/pca_dense_correlation_distr_step2_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
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
    DistributedInput<correlationDense> * input      = static_cast<DistributedInput<correlationDense> *>(_in);
    PartialResult<correlationDense> * partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    DistributedParameter<step2Master, algorithmFPType, correlationDense> * parameter =
        static_cast<DistributedParameter<step2Master, algorithmFPType, correlationDense> *>(_par);
    services::Environment::env & env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(distributed, algorithmFPType),
                                                   compute, input, partialResult, parameter);

    input->get(partialResults)->clear();
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu>::finalizeCompute()
{
    PartialResult<correlationDense> * partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    Result * result                                 = static_cast<Result *>(_res);
    DistributedParameter<step2Master, algorithmFPType, correlationDense> * parameter =
        static_cast<DistributedParameter<step2Master, algorithmFPType, correlationDense> *>(_par);

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(distributed, algorithmFPType), finalize, partialResult, parameter,
                       *eigenvectors, *eigenvalues);
}

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
