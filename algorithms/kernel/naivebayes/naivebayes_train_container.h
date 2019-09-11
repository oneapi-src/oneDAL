/* file: naivebayes_train_container.h */
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

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    const NumericTable *data   = input->get(classifier::training::data).get();
    const NumericTable *labels = input->get(classifier::training::labels).get();

    Model *model = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesBatchTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, data, labels, model, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    size_t na = input->size();

    const NumericTable *data   = input->get(classifier::training::data).get();
    const NumericTable *labels = input->get(classifier::training::labels).get();

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesOnlineTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, data, labels, pModel, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();
    Model        *rModel = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesOnlineTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, pModel, rModel, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NaiveBayesBatchTrainKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput *input = static_cast<DistributedInput *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    data_management::DataCollection *models = input->get(partialModels).get();

    size_t na = models->size();

    PartialModel **a = new PartialModel*[na];

    for(size_t i = 0; i < na; i++)
    {
        a[i] = static_cast<PartialModel *>((*models)[i].get());
    }

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::NaiveBayesDistributedTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), merge, na, a, pModel, par);

    models->clear();
    delete[] a;
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    PartialModel *pModel = partialResult->get(classifier::training::partialModel).get();
    Model        *rModel = result->get(classifier::training::model).get();

    const Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NaiveBayesDistributedTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, pModel,
                       rModel, par);
}

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
