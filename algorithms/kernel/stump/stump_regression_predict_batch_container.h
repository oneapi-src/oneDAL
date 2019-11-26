/* file: stump_regression_predict_batch_container.h */
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
//  Implementation of Decision Stump prediction algorithm container --
//  a class that contains Fast Decision Stump kernels for supported architectures.
//--
*/

#ifndef __STUMP_REGRESSION_PREDICT_BATCH_CONTAINER_H__
#define __STUMP_REGRESSION_PREDICT_BATCH_CONTAINER_H__

#include "stump_regression_predict.h"
#include "stump_regression_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace prediction
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::StumpPredictKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    daal::algorithms::regression::prediction::Input * input   = static_cast<daal::algorithms::regression::prediction::Input *>(_in);
    daal::algorithms::regression::prediction::Result * result = static_cast<daal::algorithms::regression::prediction::Result *>(_res);
    DAAL_CHECK_MALLOC(_par)
    const Parameter * par = static_cast<Parameter *>(_par);

    NumericTable * a             = static_cast<NumericTable *>(input->get(daal::algorithms::regression::prediction::data).get());
    stump::regression::Model * m = static_cast<stump::regression::Model *>(input->get(daal::algorithms::regression::prediction::model).get());

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::StumpPredictKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m,
                       result->get(daal::algorithms::regression::prediction::prediction).get(), par);
}

} // namespace prediction
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
