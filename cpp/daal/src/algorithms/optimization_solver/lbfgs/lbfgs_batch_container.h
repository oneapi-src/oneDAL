/* file: lbfgs_batch_container.h */
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
//  Implementation of LBFGS algorithm container.
//--
*/

#include "algorithms/optimization_solver/lbfgs/lbfgs_batch.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_base.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_dense_default_kernel.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace interface2
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LBFGSKernel, algorithmFPType, method);
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

    NumericTable * correctionPairsInput      = input->get(lbfgs::correctionPairs).get();
    NumericTable * correctionIndicesInput    = input->get(lbfgs::correctionIndices).get();
    NumericTable * inputArgument             = input->get(iterative_solver::inputArgument).get();
    NumericTable * averageArgLIterInput      = input->get(averageArgumentLIterations).get();
    OptionalArgument * optionalArgumentInput = input->get(iterative_solver::optionalArgument).get();

    NumericTable * correctionPairsResult      = result->get(lbfgs::correctionPairs).get();
    NumericTable * correctionIndicesResult    = result->get(correctionIndices).get();
    NumericTable * minimum                    = result->get(iterative_solver::minimum).get();
    NumericTable * nIterations                = result->get(iterative_solver::nIterations).get();
    NumericTable * averageArgLIterResult      = result->get(averageArgumentLIterations).get();
    OptionalArgument * optionalArgumentResult = result->get(iterative_solver::optionalResult).get();

    __DAAL_CALL_KERNEL(env, internal::LBFGSKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), correctionPairsInput, correctionIndicesInput, inputArgument, averageArgLIterInput,
                       optionalArgumentInput, correctionPairsResult, correctionIndicesResult, minimum, nIterations, averageArgLIterResult,
                       optionalArgumentResult, parameter, *parameter->engine);
}

} // namespace interface2
} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
