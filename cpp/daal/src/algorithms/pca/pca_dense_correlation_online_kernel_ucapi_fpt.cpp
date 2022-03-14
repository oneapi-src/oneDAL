/* file: pca_dense_correlation_online_kernel_ucapi_fpt.cpp */
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
//  Implementation of PCA Online Kernel for GPU.
//--
*/

#include "src/algorithms/pca/oneapi/pca_dense_correlation_online_kernel_ucapi.h"
#include "src/algorithms/pca/oneapi/pca_dense_correlation_online_kernel_ucapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template class PCACorrelationKernelOnlineUCAPI<DAAL_FPTYPE>;
} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
