/* file: covariance_dense_default_distr_step2_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of Covariance kernel.
//--
*/

#include "src/externals/service_profiler.h"
#include "src/algorithms/covariance/covariance_container.h"
#include "src/algorithms/covariance/oneapi/covariance_dense_distr_step2_oneapi.h"
#include "src/algorithms/covariance/oneapi/covariance_dense_distr_step2_oneapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace oneapi
{
namespace internal
{
template class CovarianceDenseDistrStep2KernelOneAPI<DAAL_FPTYPE, defaultDense>;
}
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal
