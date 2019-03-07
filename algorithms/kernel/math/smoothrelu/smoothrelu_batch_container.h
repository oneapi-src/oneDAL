/* file: smoothrelu_batch_container.h */
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
//  Implementation of smoothrelu calculation algorithm container.
//--
*/

#ifndef __SMOOTHRELU_BATCH_CONTAINER_H__
#define __SMOOTHRELU_BATCH_CONTAINER_H__

#include "math/smoothrelu.h"
#include "smoothrelu_kernel.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace smoothrelu
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SmoothReLUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SmoothReLUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input->get(data).get(), result->get(value).get());
}

} // namespace interface1
} // namespace smoothrelu
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
