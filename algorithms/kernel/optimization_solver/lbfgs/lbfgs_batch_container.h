/* file: lbfgs_batch_container.h */
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
//  Implementation of LBFGS algorithm container.
//--
*/

#include "lbfgs_batch.h"
#include "lbfgs_base.h"
#include "lbfgs_dense_default_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LBFGSKernel, algorithmFPType, method);
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

    NumericTable *correctionPairsInput      = input->get(lbfgs::correctionPairs).get();
    NumericTable *correctionIndicesInput    = input->get(lbfgs::correctionIndices).get();
    NumericTable *inputArgument             = input->get(iterative_solver::inputArgument).get();
    NumericTable *averageArgLIterInput      = input->get(averageArgumentLIterations).get();
    OptionalArgument *optionalArgumentInput = input->get(iterative_solver::optionalArgument).get();

    NumericTable *correctionPairsResult      = result->get(lbfgs::correctionPairs).get();
    NumericTable *correctionIndicesResult    = result->get(correctionIndices).get();
    NumericTable *minimum                    = result->get(iterative_solver::minimum).get();
    NumericTable *nIterations                = result->get(iterative_solver::nIterations).get();
    NumericTable *averageArgLIterResult      = result->get(averageArgumentLIterations).get();
    OptionalArgument *optionalArgumentResult = result->get(iterative_solver::optionalResult).get();

    __DAAL_CALL_KERNEL(env, internal::LBFGSKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), correctionPairsInput, correctionIndicesInput,
        inputArgument, averageArgLIterInput, optionalArgumentInput, correctionPairsResult, correctionIndicesResult, minimum, nIterations,
        averageArgLIterResult, optionalArgumentResult, parameter, *parameter->engine);
}

} // namespace interface1

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
