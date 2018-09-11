/* file: xavier_initializer_batch_container.h */
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
//  Implementation of Xavier calculation algorithm container.
//--
*/

#ifndef __XAVIER_INITIALIZER_BATCH_CONTAINER_H__
#define __XAVIER_INITIALIZER_BATCH_CONTAINER_H__

#include "neural_networks/initializers/xavier/xavier_initializer.h"
#include "xavier_initializer_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::XavierKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    initializers::Result *result = static_cast<initializers::Result *>(_res);
    xavier::Parameter *parameter = static_cast<xavier::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::XavierKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        internal::XavierInitializerTaskDescriptor(result, parameter)
    );
}
} // namespace interface1
} // namespace xavier
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
