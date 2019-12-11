/* file: outlierdetection_univariate_batch_container.h */
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
//  Implementation of Outlier Detection algorithm container.
//--
*/

#include "outlier_detection_univariate.h"
#include "outlierdetection_univariate_kernel.h"

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OutlierDetectionKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * a                  = static_cast<NumericTable *>(input->get(data).get());
    NumericTable * r                  = static_cast<NumericTable *>(result->get(weights).get());

    NumericTable * locationTable  = static_cast<NumericTable *>(input->get(InputId::location).get());
    NumericTable * scatterTable   = static_cast<NumericTable *>(input->get(InputId::scatter).get());
    NumericTable * thresholdTable = static_cast<NumericTable *>(input->get(InputId::threshold).get());

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::OutlierDetectionKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *a, *r, locationTable,
                       scatterTable, thresholdTable);
}

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal
