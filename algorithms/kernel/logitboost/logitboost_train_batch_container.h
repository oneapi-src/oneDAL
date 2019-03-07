/* file: logitboost_train_batch_container.h */
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
//  Implementation of Logit Boost container.
//--
*/

#ifndef __LOGITBOOST_TRAIN_BATCH_CONTAINER_H__
#define __LOGITBOOST_TRAIN_BATCH_CONTAINER_H__

#include "logitboost_training_batch.h"
#include "logitboost_train_friedman_kernel.h"
#include "kernel.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LogitBoostTrainKernel, method, algorithmFPType);
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
    classifier::training::Result *result = static_cast<classifier::training::Result *>(_res);

    size_t na = input->size();

    NumericTablePtr a[2];
    a[0] = services::staticPointerCast<NumericTable>(input->get(classifier::training::data));
    a[1] = services::staticPointerCast<NumericTable>(input->get(classifier::training::labels));
    logitboost::Model *r = static_cast<logitboost::Model *>(result->get(classifier::training::model).get());
    logitboost::Parameter *par = static_cast<logitboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::LogitBoostTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, r, par);
}

} // namespace daal::algorithms::logitboost::training
}
}
} // namespace daal

#endif // __LOGITBOOST_TRAIN_BATCH_CONTAINER_H__
