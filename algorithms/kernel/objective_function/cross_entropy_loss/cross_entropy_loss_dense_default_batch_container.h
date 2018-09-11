/* file: cross_entropy_loss_dense_default_batch_container.h */
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
//  Implementation of cross_entropy_loss calculation algorithm container.
//--
*/

#ifndef __CROSS_ENTROPY_LOSS_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __CROSS_ENTROPY_LOSS_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "cross_entropy_loss_batch.h"
#include "cross_entropy_loss_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace cross_entropy_loss
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::CrossEntropyLossKernel, algorithmFPType, method);
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
    NumericTable *value    = nullptr;
    NumericTable *hessian  = nullptr;
    NumericTable *gradient = nullptr;
    if(parameter->resultsToCompute & objective_function::value)
        value = result->get(objective_function::valueIdx).get();

    if(parameter->resultsToCompute & objective_function::hessian)
        hessian = result->get(objective_function::hessianIdx).get();

    if(parameter->resultsToCompute & objective_function::gradient)
        gradient = result->get(objective_function::gradientIdx).get();

    __DAAL_CALL_KERNEL(env, internal::CrossEntropyLossKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
        compute, input->get(cross_entropy_loss::data).get(), input->get(cross_entropy_loss::dependentVariables).get(), input->get(cross_entropy_loss::argument).get(),
        value, hessian, gradient, parameter);
}

} // namespace interface1

} // namespace cross_entropy_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
