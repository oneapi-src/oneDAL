/* file: logitboost_predict_batch_container.h */
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
//  Implementation of Logit Boost algorithm container -- a class
//  that contains fast Logit Boost prediction kernels
//  for supported architectures.
//--
*/

#ifndef __LOGITBOOST_PREDICT_BATCH_CONTAINER__
#define __LOGITBOOST_PREDICT_BATCH_CONTAINER__

#include "logitboost_predict.h"
#include "logitboost_predict_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LogitBoostPredictKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Input *input = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTablePtr a = input->get(classifier::prediction::data);
    logitboost::Model *m = static_cast<logitboost::Model *>(input->get(classifier::prediction::model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());
    logitboost::Parameter *par = static_cast<logitboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::LogitBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}

} // namespace daal::algorithms::logitboost::prediction
}
}
} // namespace daal

#endif // __LOGITBOOST_PREDICT_CONTAINER__
