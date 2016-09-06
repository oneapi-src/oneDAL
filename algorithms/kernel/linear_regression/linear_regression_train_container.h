/* file: linear_regression_train_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of linear regression container.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_CONTAINER_H__
#define __LINEAR_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "linear_regression_training_batch.h"
#include "linear_regression_training_online.h"
#include "linear_regression_training_distributed.h"
#include "linear_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{

/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename interm, training::Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LinearRegressionTrainBatchKernel, interm, method);
}

template <typename interm, training::Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename interm, training::Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *x = input->get(data).get();
    NumericTable *y = input->get(dependentVariables).get();

    linear_regression::Model *r = result->get(model).get();

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(interm, method),    \
            compute, x, y, r, par);
}


/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename interm, training::Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LinearRegressionTrainOnlineKernel, interm, method);
}

template <typename interm, training::Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename interm, training::Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    NumericTable *x = input->get(data).get();
    NumericTable *y = input->get(dependentVariables).get();

    linear_regression::Model *partialModel = partialResult->get(training::partialModel).get();

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionTrainOnlineKernel, __DAAL_KERNEL_ARGUMENTS(interm, method),   \
            compute, x, y, partialModel, par);
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename interm, training::Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::finalizeCompute()
{
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    linear_regression::Model *partialModel = partialResult->get(training::partialModel).get();
    linear_regression::Model *model = result->get(training::model).get();

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionTrainOnlineKernel, __DAAL_KERNEL_ARGUMENTS(interm, method),   \
            finalizeCompute, partialModel, model, par);
}


/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename interm, training::Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LinearRegressionTrainDistributedKernel, interm, method);
}

template <typename interm, training::Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename interm, training::Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::compute()
{
    DistributedInput<step2Master> *input = static_cast<DistributedInput<step2Master> *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    data_management::DataCollection *collection
        = static_cast<data_management::DataCollection*>(input->get(partialModels).get());
    size_t n = collection->size();

    daal::algorithms::Model **localModels = new daal::algorithms::Model*[n];
    for(size_t i = 0; i < n; i++)
    {
        localModels[i] = static_cast<daal::algorithms::Model *>((*collection)[i].get());
    }

    daal::algorithms::Model *partialModel = static_cast<daal::algorithms::Model *>(partialResult->get(
                                                                                       training::partialModel).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionTrainDistributedKernel, __DAAL_KERNEL_ARGUMENTS(interm, method),  \
            compute, n, localModels, partialModel, par);

    collection->clear();
    delete [] localModels;
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename interm, training::Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::finalizeCompute()
{
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    linear_regression::Model *partialModel = partialResult->get(training::partialModel).get();
    linear_regression::Model *model = result->get(training::model).get();

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionTrainDistributedKernel, __DAAL_KERNEL_ARGUMENTS(interm, method),  \
            finalizeCompute, partialModel, model, par);
}

}
}
}
}

#endif
