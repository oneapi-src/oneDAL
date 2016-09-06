/* file: em_gmm_dense_default_batch_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of EM calculation algorithm container.
//--
*/

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "em_gmm.h"
#include "em_gmm_dense_default_batch_kernel.h"

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
void BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input  *input = static_cast<Input *>(_in);
    Result *pRes   = static_cast<Result *>(_res);
    Parameter *emPar = static_cast<Parameter *>(_par);
    size_t nComponents = emPar->nComponents;

    size_t na = 1;
    size_t nr = 2 + nComponents;

    NumericTable **a = new NumericTable*[3 + nComponents];
    NumericTable **r = new NumericTable*[4 + nComponents];

    a[0] = static_cast<NumericTable *>(input->get(data).get());
    a[1] = static_cast<NumericTable *>(input->get(inputWeights).get());
    a[2] = static_cast<NumericTable *>(input->get(inputMeans).get());
    data_management::DataCollectionPtr initCovs = input->get(inputCovariances);
    for(size_t i = 0; i < nComponents; i++)
    {
        a[3 + i] = static_cast<NumericTable *>((*initCovs)[i].get());
    }

    r[0] = static_cast<NumericTable *>(pRes->get(weights).get());
    r[1] = static_cast<NumericTable *>(pRes->get(means).get());
    r[2] = static_cast<NumericTable *>(pRes->get(goalFunction).get());
    r[3] = static_cast<NumericTable *>(pRes->get(nIterations).get());
    data_management::DataCollectionPtr resultCovs = pRes->get(covariances);
    for(size_t i = 0; i < nComponents; i++)
    {
        r[4 + i] = static_cast<NumericTable *>((*resultCovs)[i].get());
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::EMKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, emPar);

    delete [] a;
    delete [] r;
}

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
