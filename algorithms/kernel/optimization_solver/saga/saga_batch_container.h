/* file: saga_batch_container.h */
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
//  Implementation of saga calculation algorithm container.
//--
*/

#ifndef __SAGA_BATCH_CONTAINER_H__
#define __SAGA_BATCH_CONTAINER_H__

#include "saga_batch.h"
#include "saga_dense_default_kernel.h"
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
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SagaKernel, algorithmFPType, method);
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
    Result *result = static_cast<Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env &env = *_env;

    NumericTable *inputArgument           = input->get(iterative_solver::inputArgument).get();

    NumericTable *minimum                 = result->get(iterative_solver::minimum).get();
    NumericTable *nIterations             = result->get(iterative_solver::nIterations).get();

    NumericTable *gradientsTableInput  = input->get(saga::gradientsTable).get();
    NumericTable *gradientsTableResult = result->get(gradientsTable).get();

    __DAAL_CALL_KERNEL(env, internal::SagaKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations,
        gradientsTableInput, gradientsTableResult, parameter, *parameter->engine);
}

} // namespace interface1

} // namespace saga

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
