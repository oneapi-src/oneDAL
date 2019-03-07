/* file: pca_dense_correlation_online_container.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_ONLINE_CONTAINER_H__
#define __PCA_DENSE_CORRELATION_ONLINE_CONTAINER_H__

#include "kernel.h"
#include "pca_online.h"
#include "pca_dense_correlation_online_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, correlationDense, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCACorrelationKernel, online, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, correlationDense, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, correlationDense, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    OnlineParameter<algorithmFPType, correlationDense> *parameter = static_cast<OnlineParameter<algorithmFPType, correlationDense> *>(_par);
    PartialResult<correlationDense> *partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    services::Environment::env &env = *_env;

    data_management::NumericTablePtr data = input->get(pca::data);

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(online, algorithmFPType),
                       compute, data, partialResult, parameter);
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, correlationDense, cpu>::finalizeCompute()
{
    Input *input = static_cast<Input *>(_in);
    OnlineParameter<algorithmFPType, correlationDense> *parameter = static_cast<OnlineParameter<algorithmFPType, correlationDense> *>(_par);
    PartialResult<correlationDense> *partialResult = static_cast<PartialResult<correlationDense> *>(_pres);
    Result *result = static_cast<Result *>(_res);
    services::Environment::env &env = *_env;

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(online, algorithmFPType),
                       finalize, partialResult, parameter, *eigenvectors, *eigenvalues);
}

}
}
} // namespace daal

#endif
