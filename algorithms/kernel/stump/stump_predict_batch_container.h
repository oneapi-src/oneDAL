/* file: stump_predict_batch_container.h */
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
//  Implementation of Decision Stump prediction algorithm container --
//  a class that contains Fast Decision Stump kernels for supported architectures.
//--
*/

#ifndef __STUMP_PREDICT_BATCH_CONTAINER_H__
#define __STUMP_PREDICT_BATCH_CONTAINER_H__

#include "stump_predict.h"
#include "stump_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace prediction
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::StumpPredictKernel, method, algorithmFPType);
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

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    stump::Model *m = static_cast<stump::Model *>(input->get(classifier::prediction::model).get());

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::StumpPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m,
        result->get(classifier::prediction::prediction).get(), NULL);
}

} // daal::algorithms::stump::prediction
}
}
} // namespace daal

#endif
