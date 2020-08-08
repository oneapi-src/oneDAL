/* file: kmeans_dense_lloyd_batch_kernel_ucapi_fpt.cpp */
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
//  Implementation of K-means Distr Step1 Kernel for GPU.
//--
*/

#include "oneapi/kmeans_lloyd_distr_step1_kernel_ucapi.h"
#include "oneapi/kmeans_lloyd_distr_step1_ucapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template class KMeansDistributedStep1KernelUCAPI<DAAL_FPTYPE>;
} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
