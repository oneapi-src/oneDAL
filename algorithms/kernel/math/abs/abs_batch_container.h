/* file: abs_batch_container.h */
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
//  Implementation of abs calculation algorithm container.
//--
*/
#ifndef __ABS_BATCH_CONTAINER_H__
#define __ABS_BATCH_CONTAINER_H__

#include "math/abs.h"
#include "abs_dense_default_kernel.h"
#include "abs_csr_fast_kernel.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace abs
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AbsKernel, algorithmFPType, method);
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

    NumericTablePtr inputTable = input->get(data);
    NumericTablePtr resultTable = result->get(value);

    __DAAL_CALL_KERNEL(env, internal::AbsKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTable.get(), resultTable.get());
}

} // namespace interface1
} // namespace abs
} // namespace math
} // namespace algorithms
} // namespace daal
#endif
