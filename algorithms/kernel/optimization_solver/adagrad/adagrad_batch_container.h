/* file: adagrad_batch_container.h */
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
//  Implementation of adagrad calculation algorithm container.
//--
*/

#ifndef __ADAGRAD_BATCH_CONTAINER_H__
#define __ADAGRAD_BATCH_CONTAINER_H__

#include "adagrad_batch.h"
#include "adagrad_dense_default_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AdagradKernel, algorithmFPType, method);
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
    NumericTable *gradientSquareSumInput  = input->get(adagrad::gradientSquareSum).get();
    OptionalArgument *optionalArgument    = input->get(iterative_solver::optionalArgument).get();

    NumericTable *minimum                 = result->get(iterative_solver::minimum).get();
    NumericTable *nIterations             = result->get(iterative_solver::nIterations).get();
    NumericTable *gradientSquareSumResult = result->get(gradientSquareSum).get();
    OptionalArgument *optionalResult      = result->get(iterative_solver::optionalResult).get();

    __DAAL_CALL_KERNEL(env, internal::AdagradKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations,
        gradientSquareSumResult, gradientSquareSumInput, optionalArgument,
        optionalResult, parameter, *parameter->engine);
}

} // namespace interface1

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
