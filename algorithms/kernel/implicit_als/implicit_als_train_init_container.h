/* file: implicit_als_train_init_container.h */
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
//  Implementation of implicit ALS initialization algorithm container.
//--
*/

#ifndef __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__
#define __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__

#include "implicit_als_training_init_batch.h"
#include "implicit_als_training_init_distributed.h"
#include "implicit_als_train_init_kernel.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : TrainingContainerIface<batch>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(data).get());
    Model *m = static_cast<Model *>(result->get(training::init::model).get());
    NumericTable *r = m->getItemsFactors().get();

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, a, r, par);
}

/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitDistrKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
void DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    NumericTable *dataTable = static_cast<NumericTable *>(input->get(data).get());


    implicit_als::PartialModel *pModel = static_cast<implicit_als::PartialModel *>(partialResult->get(partialModel).get());
    NumericTable *result = pModel->getFactors().get();

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitDistrKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, dataTable, result, par);
}

}
}
}
}
}

#endif
