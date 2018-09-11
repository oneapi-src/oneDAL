/* file: mse_dense_default_batch_container.h */
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
//  Implementation of mse calculation algorithm container.
//--
*/

#ifndef __MSE_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __MSE_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "mse_batch.h"
#include "mse_dense_default_batch_kernel.h"

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
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MSEKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    objective_function::Result *result = static_cast<objective_function::Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env &env = *_env;

    NumericTable *data               = input->get(mse::data).get();
    NumericTable *dependentVariables = input->get(mse::dependentVariables).get();
    NumericTable *argument           = input->get(mse::argument).get();

    NumericTable *value    = NULL;
    NumericTable *hessian  = NULL;
    NumericTable *gradient = NULL;

    bool valueFlag = ((parameter->resultsToCompute & objective_function::value) != 0) ? true : false;
    if(valueFlag)
    {
        value = result->get(objective_function::valueIdx).get();
    }

    bool hessianFlag = ((parameter->resultsToCompute & objective_function::hessian) != 0) ? true : false;
    if(hessianFlag)
    {
        hessian = result->get(objective_function::hessianIdx).get();
    }

    bool gradientFlag = ((parameter->resultsToCompute & objective_function::gradient) != 0) ? true : false;
    if (gradientFlag)
    {
        gradient = result->get(objective_function::gradientIdx).get();
    }

    __DAAL_CALL_KERNEL(env, internal::MSEKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, data, dependentVariables, argument, value,
                                                                         hessian, gradient, parameter);
}

} // namespace interface1

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
