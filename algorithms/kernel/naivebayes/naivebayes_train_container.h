/* file: naivebayes_train_container.h */
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
//  Implementation of Naive Bayes algorithm container -- a class that contains
//  Naive Bayes kernels for supported architectures.
//--
*/

#include "multinomial_naive_bayes_training_batch.h"
#include "multinomial_naive_bayes_training_online.h"
#include "multinomial_naive_bayes_training_distributed.h"
#include "multinomial_naive_bayes_training_types.h"
#include "naivebayes_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    size_t na = input->size();

    NumericTable *a[2];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());

    Model *r = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesBatchTrainKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, r, par);
}

template<typename interm, Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    size_t na = input->size();

    NumericTable *a[2];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesOnlineTrainKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, pModel, par);
}

template<typename interm, Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();
    Model        *rModel = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesOnlineTrainKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), finalizeCompute, pModel, rModel,
                       par);
}

template<typename interm, Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::compute()
{
    classifier::training::DistributedInput *input = static_cast<classifier::training::DistributedInput *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    data_management::DataCollection *models = input->get(classifier::training::partialModels).get();

    size_t na = models->size();

    PartialModel **a = new PartialModel*[na];

    for(size_t i = 0; i < na; i++)
    {
        a[i] = static_cast<PartialModel *>((*models)[i].get());
    }

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesDistributedTrainKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), merge, na, a, pModel, par);

    models->clear();
    delete[] a;
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();
    Model        *rModel = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesDistributedTrainKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), finalizeCompute, pModel,
                       rModel, par);
}

} // namespace training

} // namespace multinomial_naive_bayes

} // namespace algorithms

} // namespace daal
