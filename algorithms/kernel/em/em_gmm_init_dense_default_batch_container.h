/* file: em_gmm_init_dense_default_batch_container.h */
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
//  Implementation of EM calculation algorithm container.
//--
*/

#ifndef __EM_GMM_INIT_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __EM_GMM_INIT_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "em_gmm_init_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{

/**
 *  \brief Initialize list of em default init kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EMInitKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input  *input = static_cast<Input *>(_in);
    Result *pRes   = static_cast<Result *>(_res);

    Parameter *emPar = static_cast<Parameter *>(_par);

    NumericTable* inputData        = input->get(data).get();
    NumericTable* inputWeights     = pRes->get(weights).get();
    NumericTable* inputMeans       = pRes->get(means).get();
    data_management::DataCollectionPtr inputCovariances = pRes->get(covariances);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::EMInitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, *inputData, *inputWeights, *inputMeans, inputCovariances, *emPar, *emPar->engine);

}

} // namespace init

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
