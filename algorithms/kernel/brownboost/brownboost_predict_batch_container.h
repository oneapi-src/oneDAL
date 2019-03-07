/* file: brownboost_predict_batch_container.h */
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
//  Implementation of Brown Boost prediction algorithm container --
//  a class that contains Fast Brown Boost kernels for supported architectures.
//--
*/

#include "brownboost_predict.h"
#include "brownboost_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BrownBoostPredictKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);
    classifier::prediction::Input *input = static_cast<classifier::prediction::Input *>(_in);

    NumericTablePtr a = input->get(classifier::prediction::data);
    brownboost::Model *m = static_cast<brownboost::Model *>(input->get(classifier::prediction::model).get());
    NumericTablePtr r = result->get(classifier::prediction::prediction);
    brownboost::Parameter *par = static_cast<brownboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BrownBoostPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, r, par);
}

} // namespace daal::algorithms::brownboost::prediction
}
}
} // namespace daal
