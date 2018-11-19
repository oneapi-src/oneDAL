/* file: logistic_regression_train_container.h */
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
//  Implementation of logistic regression container.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_CONTAINER_H__
#define __LOGISTIC_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "algorithms/logistic_regression/logistic_regression_training_batch.h"
#include "logistic_regression_train_kernel.h"
#include "logistic_regression_model_impl.h"
#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    auto x = input->get(classifier::training::data);
    auto y = input->get(classifier::training::labels);
    logistic_regression::Model *m = result->get(classifier::training::model).get();
    const logistic_regression::training::Parameter *par = static_cast<logistic_regression::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::TrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
        compute, daal::services::internal::getHostApp(*input), x, y, *m, *result, *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result *result = static_cast<Result *>(_res);
    logistic_regression::Model *m = result->get(classifier::training::model).get();
    logistic_regression::internal::ModelImpl* pImpl = dynamic_cast<logistic_regression::internal::ModelImpl*>(m);

    logistic_regression::training::Parameter *par = static_cast<logistic_regression::training::Parameter*>(_par);
    if(!par->optimizationSolver.get())
    {
        auto solver = optimization_solver::sgd::Batch<algorithmFPType, optimization_solver::sgd::momentum>::create();
        par->optimizationSolver = solver;
        const size_t nIterations = 1000;
        const algorithmFPType  learningRate = 1e-3;
        const algorithmFPType accuracyThreshold = 1e-4;
        solver->parameter.learningRateSequence = HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, learningRate);
        solver->parameter.accuracyThreshold = accuracyThreshold;
        solver->parameter.nIterations = nIterations;
        classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
        solver->parameter.batchSize = input->get(classifier::training::data)->getNumberOfRows();
    }
    DAAL_ASSERT(pImpl);
    return pImpl->reset(par->interceptFlag);
}

}
}
}
}
#endif
