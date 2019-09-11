/* file: coordinate_descent_batch_container.h */
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
//  Implementation of coordinate_descent calculation algorithm container.
//--
*/

#ifndef __COORDINATE_DESCENT_BATCH_CONTAINER_H__
#define __COORDINATE_DESCENT_BATCH_CONTAINER_H__

#include "coordinate_descent_batch.h"
#include "coordinate_descent_dense_default_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace coordinate_descent
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::CoordinateDescentKernel, algorithmFPType, method);
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

    optimization_solver::objective_function::ResultPtr hesGrResultPtr = optimization_solver::objective_function::ResultPtr(new optimization_solver::objective_function::Result());
    optimization_solver::objective_function::ResultPtr proxResultPtr = optimization_solver::objective_function::ResultPtr(new optimization_solver::objective_function::Result());

    __DAAL_CALL_KERNEL(env, internal::CoordinateDescentKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations, parameter, *parameter->engine, hesGrResultPtr, proxResultPtr);
}

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
