/* file: em_gmm_dense_default_batch_container.h */
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

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "em_gmm.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{

/**
 *  \brief Initialize list of em kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EMKernel, algorithmFPType, method);
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
    const Parameter *emPar = static_cast<Parameter *>(_par);
    size_t nComponents = emPar->nComponents;

    NumericTable *dataTable = input->get(data).get();
    NumericTable *initialWeights = input->get(inputWeights).get();
    NumericTable *initialMeans = input->get(inputMeans).get();
    daal::internal::TArray<NumericTable *, cpu> initialCovariancesPtr(nComponents);
    NumericTable **initialCovariances = initialCovariancesPtr.get();
    for(size_t i = 0; i < nComponents; i++)
    {
        initialCovariances[i] = input->get(inputCovariances, i).get();
    }

    NumericTable *resultWeights = pRes->get(weights).get();
    NumericTable *resultMeans = pRes->get(means).get();
    NumericTable *resultGoalFunction = pRes->get(goalFunction).get();
    NumericTable *resultNIterations = pRes->get(nIterations).get();

    daal::internal::TArray<NumericTable *, cpu> resultCovariancesPtr(nComponents);
    NumericTable **resultCovariances = resultCovariancesPtr.get();
    for(size_t i = 0; i < nComponents; i++)
    {
        resultCovariances[i] = pRes->get(covariances, i).get();
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::EMKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *dataTable, *initialWeights, *initialMeans, initialCovariances, *resultWeights, *resultMeans, resultCovariances, *resultNIterations, *resultGoalFunction, *emPar)

}

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
