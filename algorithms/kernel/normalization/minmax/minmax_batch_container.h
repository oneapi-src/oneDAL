/* file: minmax_batch_container.h */
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
//  Implementation of minmax calculation algorithm container.
//--
*/

#ifndef __MINMAX_BATCH_CONTAINER_H__
#define __MINMAX_BATCH_CONTAINER_H__

#include "normalization/minmax.h"
#include "minmax_moments.h"
#include "minmax_kernel.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MinMaxKernel, algorithmFPType, method);
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
    Parameter<algorithmFPType> *parameter = static_cast<Parameter<algorithmFPType> *>(_par);

    NumericTablePtr dataTable = input->get(data);
    NumericTablePtr normalizedDataTable = result->get(normalizedData);
    low_order_moments::BatchImpl *moments = parameter->moments.get();

    NumericTablePtr minimums;
    NumericTablePtr maximums;
    Status s;
    DAAL_CHECK_STATUS(s, internal::computeMinimumsAndMaximums(moments, dataTable, minimums, maximums));

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MinMaxKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       *dataTable.get(), *normalizedDataTable.get(), *minimums.get(), *maximums.get(),
                       (algorithmFPType)(parameter->lowerBound), (algorithmFPType)(parameter->upperBound));
}

} // namespace interface1
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
