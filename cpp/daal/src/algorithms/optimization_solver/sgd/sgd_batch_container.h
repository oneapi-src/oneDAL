/* file: sgd_batch_container.h */
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
//  Implementation of sgd calculation algorithm container.
//--
*/

#ifndef __SGD_BATCH_CONTAINER_H__
#define __SGD_BATCH_CONTAINER_H__

#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "src/algorithms/optimization_solver/sgd/sgd_dense_default_kernel.h"
#include "src/algorithms/optimization_solver/sgd/sgd_dense_minibatch_kernel.h"
#include "src/algorithms/optimization_solver/sgd/sgd_dense_momentum_kernel.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface2
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu || method == defaultDense || method == momentum)
    {
        __DAAL_INITIALIZE_KERNELS(internal::SGDKernel, algorithmFPType, method);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::SGDKernelOneAPI, algorithmFPType, method);
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
    iterative_solver::Input * input   = static_cast<iterative_solver::Input *>(_in);
    iterative_solver::Result * result = static_cast<iterative_solver::Result *>(_res);
    Parameter<method> * parameter     = static_cast<Parameter<method> *>(_par);

    daal::services::Environment::env & env = *_env;

    NumericTable * inputArgument        = input->get(iterative_solver::inputArgument).get();
    NumericTablePtr minimum             = result->get(iterative_solver::minimum);
    NumericTable * nIterations          = result->get(iterative_solver::nIterations).get();
    OptionalArgument * optionalArgument = input->get(iterative_solver::optionalArgument).get();
    OptionalArgument * optionalResult   = result->get(iterative_solver::optionalResult).get();

    NumericTable * learningRateSequence = parameter->learningRateSequence.get();
    NumericTable * batchIndices         = parameter->batchIndices.get();

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu || method == defaultDense || method == momentum)
    {
        __DAAL_CALL_KERNEL(env, internal::SGDKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                           daal::services::internal::hostApp(*input), inputArgument, minimum.get(), nIterations, parameter, learningRateSequence,
                           batchIndices, optionalArgument, optionalResult, *parameter->engine);
    }
    else
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::SGDKernelOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations, parameter, learningRateSequence,
                                batchIndices, optionalArgument, optionalResult, *parameter->engine);
    }
}

} // namespace interface2

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
