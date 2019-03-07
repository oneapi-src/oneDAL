/* file: sgd_batch_container.h */
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
//  Implementation of sgd calculation algorithm container.
//--
*/

#ifndef __SGD_BATCH_CONTAINER_H__
#define __SGD_BATCH_CONTAINER_H__

#include "sgd_batch.h"
#include "sgd_dense_default_kernel.h"
#include "sgd_dense_minibatch_kernel.h"
#include "sgd_dense_momentum_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SGDKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    iterative_solver::Input *input = static_cast<iterative_solver::Input *>(_in);
    iterative_solver::Result *result = static_cast<iterative_solver::Result *>(_res);
    Parameter<method> *parameter = static_cast<Parameter<method> *>(_par);

    daal::services::Environment::env &env = *_env;

    NumericTable *inputArgument = input->get(iterative_solver::inputArgument).get();
    NumericTable *minimum       = result->get(iterative_solver::minimum).get();
    NumericTable *nIterations   = result->get(iterative_solver::nIterations).get();
    OptionalArgument *optionalArgument = input->get(iterative_solver::optionalArgument).get();
    OptionalArgument *optionalResult = result->get(iterative_solver::optionalResult).get();

    NumericTable *learningRateSequence = parameter->learningRateSequence.get();
    NumericTable *batchIndices         = parameter->batchIndices.get();

    __DAAL_CALL_KERNEL(env, internal::SGDKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
        compute, daal::services::internal::hostApp(*input), inputArgument, minimum, nIterations, parameter,
        learningRateSequence, batchIndices, optionalArgument, optionalResult, *parameter->engine);
}

} // namespace interface1

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
