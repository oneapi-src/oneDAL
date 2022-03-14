/* file: covariance_csr_singlepass_online_fpt_cpu.cpp */
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
//  Implementation of Covariance kernel.
//--
*/

#include "src/algorithms/covariance/covariance_container.h"
#include "src/algorithms/covariance/covariance_csr_online_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
template class OnlineContainer<DAAL_FPTYPE, singlePassCSR, DAAL_CPU>;
}
namespace internal
{
template class CovarianceCSROnlineKernel<DAAL_FPTYPE, singlePassCSR, DAAL_CPU>;
}
} // namespace covariance
} // namespace algorithms
} // namespace daal
