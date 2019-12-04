/* file: quantiles_batch_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of Covariance algorithm container.
//--
*/

#ifndef __QUANTILES_BATCH_CONTAINER_H__
#define __QUANTILES_BATCH_CONTAINER_H__

#include "quantiles_batch.h"
#include "quantiles_kernel.h"
#include "kernel.h"
#include "homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace quantiles
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::QuantilesKernel, defaultDense, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result * result = static_cast<Result *>(_res);
    Input * input   = static_cast<Input *>(_in);
    Parameter * par = static_cast<Parameter *>(_par);

    NumericTable * dataTable           = static_cast<NumericTable *>(input->get(data).get());
    NumericTable * quantilesTable      = static_cast<NumericTable *>(result->get(quantiles).get());
    NumericTable * quantileOrdersTable = par->quantileOrders.get();

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::QuantilesKernel, __DAAL_KERNEL_ARGUMENTS(defaultDense, algorithmFPType), compute, *dataTable,
                       *quantileOrdersTable, *quantilesTable);
}

} // namespace quantiles

} // namespace algorithms

} // namespace daal

#endif
