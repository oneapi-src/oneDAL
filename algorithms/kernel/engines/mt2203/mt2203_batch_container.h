/* file: mt2203_batch_container.h */
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
//  Implementation of mt2203 calculation algorithm container.
//--
*/

#ifndef __MT2203_BATCH_CONTAINER_H__
#define __MT2203_BATCH_CONTAINER_H__

#include "engines/mt2203/mt2203.h"
#include "mt2203_kernel.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt2203
{
namespace interface1
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::Mt2203Kernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    daal::services::Environment::env &env = *_env;
    engines::Result *result   = static_cast<engines::Result *>(_res);
    NumericTable *resultTable = result->get(engines::randomNumbers).get();

    __DAAL_CALL_KERNEL(env, internal::Mt2203Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, resultTable);
}

} // namespace interface1
} // namespace mt2203
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
