/* file: em_gmm_dense_default_batch_container.h */
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
//  Implementation of EM calculation algorithm container.
//--
*/

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "em_gmm.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
/**
 *  \brief Initialize list of em kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EMKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input           = static_cast<Input *>(_in);
    Result * pRes           = static_cast<Result *>(_res);
    const Parameter * emPar = static_cast<Parameter *>(_par);
    size_t nComponents      = emPar->nComponents;

    NumericTable * dataTable      = input->get(data).get();
    NumericTable * initialWeights = input->get(inputWeights).get();
    NumericTable * initialMeans   = input->get(inputMeans).get();
    daal::internal::TArray<NumericTable *, cpu> initialCovariancesPtr(nComponents);
    NumericTable ** initialCovariances = initialCovariancesPtr.get();
    for (size_t i = 0; i < nComponents; i++)
    {
        initialCovariances[i] = input->get(inputCovariances, i).get();
    }

    NumericTable * resultWeights      = pRes->get(weights).get();
    NumericTable * resultMeans        = pRes->get(means).get();
    NumericTable * resultGoalFunction = pRes->get(goalFunction).get();
    NumericTable * resultNIterations  = pRes->get(nIterations).get();

    daal::internal::TArray<NumericTable *, cpu> resultCovariancesPtr(nComponents);
    NumericTable ** resultCovariances = resultCovariancesPtr.get();
    for (size_t i = 0; i < nComponents; i++)
    {
        resultCovariances[i] = pRes->get(covariances, i).get();
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::EMKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *dataTable, *initialWeights, *initialMeans,
                       initialCovariances, *resultWeights, *resultMeans, resultCovariances, *resultNIterations, *resultGoalFunction, *emPar)
}

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
