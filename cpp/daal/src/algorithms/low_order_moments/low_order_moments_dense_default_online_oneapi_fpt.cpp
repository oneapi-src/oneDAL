/* file: low_order_moments_dense_default_online_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of low order moments kernel.
//--
*/

#include "src/externals/service_profiler.h"

#include "src/algorithms/low_order_moments/low_order_moments_container.h"
#include "src/algorithms/low_order_moments/oneapi/low_order_moments_online_oneapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace oneapi
{
namespace internal
{
template class DAAL_EXPORT LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, defaultDense>;
template class LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, singlePassDense>;
template class LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, sumDense>;
template class LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, fastCSR>;
template class LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, singlePassCSR>;
template class LowOrderMomentsOnlineKernelOneAPI<DAAL_FPTYPE, sumCSR>;
} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
