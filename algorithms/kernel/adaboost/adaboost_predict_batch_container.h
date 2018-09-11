/* file: adaboost_predict_batch_container.h */
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
//  Implementation of Ada Boost prediction algorithm container --
//  a class that contains Fast Ada Boost kernels for supported architectures.
//--
*/

#ifndef __ADABOOST_PREDICT_BATCH_CONTAINER__
#define __ADABOOST_PREDICT_BATCH_CONTAINER__

#include "adaboost_predict.h"
#include "adaboost_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace prediction
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AdaBoostPredictKernel, method, algorithmFPType);
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
    adaboost::Model *m = static_cast<adaboost::Model *>(input->get(classifier::prediction::model).get());
    NumericTablePtr r = result->get(classifier::prediction::prediction);
    adaboost::Parameter *par = static_cast<adaboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AdaBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}

} // namespace daal::algorithms::adaboost::prediction
}
}
} // namespace daal

#endif // __ADABOOST_PREDICT_BATCH_CONTAINER__
