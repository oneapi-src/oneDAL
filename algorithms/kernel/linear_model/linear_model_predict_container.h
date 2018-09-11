/* file: linear_model_predict_container.h */
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
//  Implementation of linear regression algorithm container -- a class
//  that contains fast linear regression prediction kernels
//  for supported architectures.
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_CONTAINER_H__
#define __LINEAR_MODEL_PREDICT_CONTAINER_H__

#include "linear_model_predict.h"
#include "linear_model_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(data).get());
    linear_model::Model *m    = static_cast<linear_model::Model    *>(input->get(model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(prediction).get());

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a, m, r);
}

}
}
}
} // namespace daal
#endif
