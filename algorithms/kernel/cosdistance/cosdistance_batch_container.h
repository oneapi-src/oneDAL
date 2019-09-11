/* file: cosdistance_batch_container.h */
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
//  Implementation of cosine distance calculation algorithm container.
//--
*/

#ifndef __COSDISTANCE_BATCH_CONTAINER_H__
#define __COSDISTANCE_BATCH_CONTAINER_H__

#include "cosine_distance.h"
#include "cosdistance_kernel.h"

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{

/**
 *  \brief Initialize list of correlation distance, double precission
 *  kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistanceKernel, algorithmFPType, method);
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
    Input *input = static_cast<Input *>(_in);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;
    NumericTable *r0 = static_cast<NumericTable *>(result->get(cosineDistance).get());
    NumericTable **r = &r0;
    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DistanceKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal

#endif
