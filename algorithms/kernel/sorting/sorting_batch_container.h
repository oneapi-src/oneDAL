/* file: sorting_batch_container.h */
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
//  Implementation of sorting algorithm container.
//--
*/

#ifndef __SORTING_BATCH_CONTAINER_H__
#define __SORTING_BATCH_CONTAINER_H__

#include "sorting_batch.h"
#include "sorting_kernel.h"
#include "kernel.h"
#include "homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace sorting
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{

    __DAAL_INITIALIZE_KERNELS(internal::SortingKernel, defaultDense, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result *result = static_cast<Result *>(_res);
    Input *input   = static_cast<Input *>(_in);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::SortingKernel, __DAAL_KERNEL_ARGUMENTS(defaultDense, algorithmFPType), compute, *(input->get(data).get()), *(result->get(sortedData).get()));
}

} // namespace daal::algorithms::sorting

} // namespace daal::algorithms

} // namespace daal

#endif
