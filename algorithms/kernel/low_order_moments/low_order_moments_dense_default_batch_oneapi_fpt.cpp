/* file: low_order_moments_dense_default_batch_oneapi_fpt.cpp */
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
//  Implementation of low order moments kernel.
//--
*/

#include "service_ittnotify.h"
DAAL_ITTNOTIFY_DOMAIN(low_order_moments.dense.batch.oneapi);

#include "low_order_moments_container.h"
#include "oneapi/low_order_moments_batch_oneapi_impl.i"

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
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, defaultDense>;
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, singlePassDense>;
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, sumDense>;
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, fastCSR>;
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, singlePassCSR>;
template class LowOrderMomentsBatchKernelOneAPI<DAAL_FPTYPE, sumCSR>;
} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
