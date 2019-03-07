/* file: kmeans_init_batch_container_v1.h */
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
//  Implementation of initializing the K-Means algorithm container.
//--
*/

#ifndef __KMEANS_INIT_CONTAINER_V1_H__
#define __KMEANS_INIT_CONTAINER_V1_H__

#include "kmeans/inner/kmeans_init_batch_v1.h"
#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{

namespace interface1
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitKernel, method, algorithmFPType);
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

    const size_t na = 1;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());

    const size_t nr = 1;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    interface1::Parameter *par = static_cast<interface1::Parameter *>(_par);
    internal::Parameter internalPar;

    internalPar.nClusters = par->nClusters;
    internalPar.seed = par->seed;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.nTrials = 1;

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansInitKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r,
        internalPar, *par->engine);
}

} // interface1

} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal

#endif
