/* file: adagrad_batch_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of adagrad calculation algorithm container.
//--
*/

#ifndef __ADAGRAD_BATCH_CONTAINER_H__
#define __ADAGRAD_BATCH_CONTAINER_H__

#include "algorithms/optimization_solver/adagrad/adagrad_batch.h"
#include "src/algorithms/optimization_solver/adagrad/adagrad_dense_default_kernel.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AdagradKernel, algorithmFPType, method);
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

    NumericTable * inputArgument          = input->get(iterative_solver::inputArgument).get();
    NumericTable * gradientSquareSumInput = input->get(adagrad::gradientSquareSum).get();
    OptionalArgument * optionalArgument   = input->get(iterative_solver::optionalArgument).get();

    NumericTable * minimum                 = result->get(iterative_solver::minimum).get();
    NumericTable * nIterations             = result->get(iterative_solver::nIterations).get();
    NumericTable * gradientSquareSumResult = result->get(gradientSquareSum).get();
    OptionalArgument * optionalResult      = result->get(iterative_solver::optionalResult).get();

    __DAAL_CALL_KERNEL(env, internal::AdagradKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations, gradientSquareSumResult,
                       gradientSquareSumInput, optionalArgument, optionalResult, parameter, *parameter->engine);
}

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
