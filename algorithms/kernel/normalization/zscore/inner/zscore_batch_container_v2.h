/* file: zscore_batch_container_v2.h */
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
//  Implementation of zscore normalization calculation algorithm container.
//--
*/

#include "zscore_v2.h"
#include "zscore_base.h"
#include "zscore_dense_default_kernel.h"
#include "zscore_dense_sum_kernel.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface2
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ZScoreKernel, algorithmFPType, method);
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
    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    NumericTablePtr inputTable      = input->get(data);
    NumericTablePtr resultTable     = result->get(normalizedData);
    NumericTablePtr resultMeans     = result->get(means);
    NumericTablePtr resultVariances = result->get(variances);

    if (method == defaultDense)
    {
        interface2::Parameter<algorithmFPType, defaultDense> * parameter = static_cast<interface2::Parameter<algorithmFPType, defaultDense> *>(par);
        interface3::Parameter<algorithmFPType, defaultDense> internalPar;
        internalPar.resultsToCompute = parameter->resultsToCompute;
        internalPar.moments          = parameter->moments;
        internalPar.doScale          = true;

        internalPar.moments->input.set(low_order_moments::data, inputTable);
        __DAAL_CALL_KERNEL(env, internal::ZScoreKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, defaultDense), compute, *inputTable, *resultTable,
                           *resultMeans, *resultVariances, internalPar);
    }
    else
    {
        interface2::Parameter<algorithmFPType, sumDense> * parameter = static_cast<interface2::Parameter<algorithmFPType, sumDense> *>(par);
        interface3::Parameter<algorithmFPType, sumDense> internalPar;
        internalPar.resultsToCompute = parameter->resultsToCompute;
        internalPar.doScale          = true;

        __DAAL_CALL_KERNEL(env, internal::ZScoreKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, sumDense), compute, *inputTable, *resultTable,
                           *resultMeans, *resultVariances, internalPar);
    }
}
} // namespace interface2

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
