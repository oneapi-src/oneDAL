/* file: mse_dense_default_batch_container_v1.h */
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
//  Implementation of mse calculation algorithm container.
//--
*/

#ifndef __MSE_DENSE_DEFAULT_BATCH_CONTAINER_V1_H__
#define __MSE_DENSE_DEFAULT_BATCH_CONTAINER_V1_H__

#include "mse_batch.h"
#include "mse_dense_default_batch_kernel_v1.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::I1MSEKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                       = static_cast<Input *>(_in);
    objective_function::Result * result = static_cast<objective_function::Result *>(_res);
    interface1::Parameter * parameter   = static_cast<interface1::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    NumericTable * data               = input->get(mse::data).get();
    NumericTable * dependentVariables = input->get(mse::dependentVariables).get();
    NumericTable * argument           = input->get(mse::argument).get();

    NumericTable * value    = nullptr;
    NumericTable * hessian  = nullptr;
    NumericTable * gradient = nullptr;

    NumericTable * nonSmoothTermValue = nullptr;
    NumericTable * proximalProjection = nullptr;
    NumericTable * lipschitzConstant  = nullptr;

    NumericTable * componentOfGradient           = nullptr;
    NumericTable * componentOfHessianDiagonal    = nullptr;
    NumericTable * componentOfProximalProjection = nullptr;

    bool valueFlag = ((parameter->resultsToCompute & objective_function::value) != 0) ? true : false;
    if (valueFlag)
    {
        value = result->get(objective_function::valueIdx).get();
    }

    bool hessianFlag = ((parameter->resultsToCompute & objective_function::hessian) != 0) ? true : false;
    if (hessianFlag)
    {
        hessian = result->get(objective_function::hessianIdx).get();
    }

    bool gradientFlag = ((parameter->resultsToCompute & objective_function::gradient) != 0) ? true : false;
    if (gradientFlag)
    {
        gradient = result->get(objective_function::gradientIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::nonSmoothTermValue)
    {
        nonSmoothTermValue = result->get(objective_function::nonSmoothTermValueIdx).get();
    }
    if (parameter->resultsToCompute & objective_function::proximalProjection)
    {
        proximalProjection = result->get(objective_function::proximalProjectionIdx).get();
    }
    if (parameter->resultsToCompute & objective_function::lipschitzConstant)
    {
        lipschitzConstant = result->get(objective_function::lipschitzConstantIdx).get();
    }

    __DAAL_CALL_KERNEL(env, internal::I1MSEKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, data, dependentVariables, argument,
                       value, hessian, gradient, nonSmoothTermValue, proximalProjection, lipschitzConstant, parameter);
}

} // namespace interface1

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
