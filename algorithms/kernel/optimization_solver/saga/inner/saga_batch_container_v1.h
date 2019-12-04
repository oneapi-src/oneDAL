/* file: saga_batch_container_v1.h */
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
//  Implementation of saga calculation algorithm container.
//--
*/

#ifndef __SAGA_BATCH_CONTAINER_V1_H__
#define __SAGA_BATCH_CONTAINER_V1_H__

#include "saga_batch.h"
#include "saga_dense_default_kernel_v1.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace saga
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::I1SagaKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input         = static_cast<Input *>(_in);
    Result * result       = static_cast<Result *>(_res);
    Parameter * parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    NumericTable * inputArgument = input->get(iterative_solver::inputArgument).get();

    NumericTable * minimum     = result->get(iterative_solver::minimum).get();
    NumericTable * nIterations = result->get(iterative_solver::nIterations).get();

    NumericTable * gradientsTableInput  = input->get(saga::gradientsTable).get();
    NumericTable * gradientsTableResult = result->get(gradientsTable).get();

    __DAAL_CALL_KERNEL(env, internal::I1SagaKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations, gradientsTableInput, gradientsTableResult,
                       parameter, *parameter->engine);
}

} // namespace interface1
} // namespace saga

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
