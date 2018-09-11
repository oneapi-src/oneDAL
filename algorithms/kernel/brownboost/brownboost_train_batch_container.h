/* file: brownboost_train_batch_container.h */
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
//  Implementation of Brown Boost algorithm container -- a class that contains
//  Freund Brown Boost kernels for supported architectures.
//--
*/

#ifndef __BROWNBOOST_TRAIN_BATCH_CONTAINER_H__
#define __BROWNBOOST_TRAIN_BATCH_CONTAINER_H__

#include "brownboost_training_batch.h"
#include "brownboost_train_kernel.h"
#include "kernel.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BrownBoostTrainKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    brownboost::training::Result *result = static_cast<brownboost::training::Result *>(_res);
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);

    size_t n = input->size();

    NumericTablePtr a[2];
    a[0] = services::staticPointerCast<NumericTable>(input->get(classifier::training::data));
    a[1] = services::staticPointerCast<NumericTable>(input->get(classifier::training::labels));
    brownboost::Model *r = static_cast<brownboost::Model *>(result->get(classifier::training::model).get());
    brownboost::Parameter *par = static_cast<brownboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BrownBoostTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, n, a, r, par);
}

}
}
}
} // namespace daal

#endif // __BROWNBOOST_TRAINING_BATCH_CONTAINER_H__
