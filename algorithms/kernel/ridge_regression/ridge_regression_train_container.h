/* file: ridge_regression_train_container.h */
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
//  Implementation of ridge regression container.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_CONTAINER_H__
#define __RIDGE_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "ridge_regression_training_batch.h"
#include "ridge_regression_training_online.h"
#include "ridge_regression_training_distributed.h"
#include "ridge_regression_train_kernel.h"
#include "ridge_regression_ne_model.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * const input = static_cast<Input *>(_in);
    Result * const result = static_cast<Result *>(_res);
    TrainParameter * const par = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr m = ridge_regression::ModelNormEq::cast(result->get(model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),    \
            compute, *(input->get(data)), *(input->get(dependentVariables)), *(m->getXTXTable()),       \
            *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag, *(par->ridgeParameters));
}

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    Input * const input = static_cast<Input *>(_in);
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    TrainParameter * const par = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr m = ridge_regression::ModelNormEq::cast(
        partialResult->get(training::partialModel));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),   \
            compute, *(input->get(data)), *(input->get(dependentVariables)), *(m->getXTXTable()),       \
            *(m->getXTYTable()), par->interceptFlag);
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    Result * const result = static_cast<Result *>(_res);
    TrainParameter * const par = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(
        partialResult->get(training::partialModel));
    ridge_regression::ModelNormEqPtr m  = ridge_regression::ModelNormEq::cast(
        result->get(training::model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),   \
            finalizeCompute, *(pm->getXTXTable()), *(pm->getXTYTable()), *(m->getXTXTable()),           \
            *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag, *(par->ridgeParameters));
}

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistributedKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> * const input = static_cast<DistributedInput<step2Master> *>(_in);
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);

    DataCollectionPtr collection = input->get(partialModels);
    size_t n = collection->size();
    TArray<NumericTable *, sse2> partialxtx(n);
    TArray<NumericTable *, sse2> partialxty(n);
    for(size_t i = 0; i < n; i++)
    {
        ridge_regression::ModelNormEq *m = static_cast<ridge_regression::ModelNormEq *>(
            (*collection)[i].get());
        partialxtx[i] = m->getXTXTable().get();
        partialxty[i] = m->getXTYTable().get();
    }

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(
            partialResult->get(training::partialModel));

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),  \
                       compute, n, partialxtx.get(), partialxty.get(),  \
                       *(pm->getXTXTable()), *(pm->getXTYTable()));

    collection->clear();
    return s;
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    Result * const result = static_cast<Result *>(_res);
    TrainParameter * const par = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(
        partialResult->get(training::partialModel));
    ridge_regression::ModelNormEqPtr m  = ridge_regression::ModelNormEq::cast(
        result->get(training::model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),  \
            finalizeCompute, *(pm->getXTXTable()), *(pm->getXTYTable()), *(m->getXTXTable()),               \
            *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag, *(par->ridgeParameters));
}

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
