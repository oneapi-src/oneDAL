/* file: uniform_batch_container.h */
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
//  Implementation of uniform algorithm container.
//--
*/

#ifndef __UNIFORM_BATCH_CONTAINER_H__
#define __UNIFORM_BATCH_CONTAINER_H__

#include "distributions/uniform/uniform.h"
#include "uniform_kernel.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::UniformKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    uniform::Parameter<algorithmFPType> *parameter = static_cast<uniform::Parameter<algorithmFPType> *>(_par);
    daal::services::Environment::env &env = *_env;

    distributions::Result *result = static_cast<distributions::Result *>(_res);

    result->set(distributions::randomNumbers, static_cast<const distributions::Input *>(_in)->get(distributions::tableToFill));
    NumericTable *resultTable = result->get(distributions::randomNumbers).get();

    __DAAL_CALL_KERNEL(env, internal::UniformKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *parameter, *parameter->engine, resultTable);
}
} // namespace interface1
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
