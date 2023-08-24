/* file: minmax_batch_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of minmax calculation algorithm container.
//--
*/

#ifndef __MINMAX_BATCH_CONTAINER_H__
#define __MINMAX_BATCH_CONTAINER_H__

#include "algorithms/normalization/minmax.h"
#include "src/algorithms/normalization/minmax/minmax_moments.h"
#include "src/algorithms/normalization/minmax/minmax_kernel.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MinMaxKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                          = static_cast<Input *>(_in);
    Result * result                        = static_cast<Result *>(_res);
    Parameter<algorithmFPType> * parameter = static_cast<Parameter<algorithmFPType> *>(_par);

    NumericTablePtr dataTable              = input->get(data);
    NumericTablePtr normalizedDataTable    = result->get(normalizedData);
    low_order_moments::BatchImpl * moments = parameter->moments.get();

    NumericTablePtr minimums;
    NumericTablePtr maximums;
    Status s;
    DAAL_CHECK_STATUS(s, internal::computeMinimumsAndMaximums(moments, dataTable, minimums, maximums));

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MinMaxKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *dataTable.get(),
                       *normalizedDataTable.get(), *minimums.get(), *maximums.get(), (algorithmFPType)(parameter->lowerBound),
                       (algorithmFPType)(parameter->upperBound));
}

} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
