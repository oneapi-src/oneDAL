/* file: kmeans_init_dense_batch_kernel_ucapi_fpt.cpp */
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
//  Implementation of K-means initialization Batch Kernel for GPU.
//--
*/

#include "oneapi/kmeans_init_dense_batch_kernel_ucapi.h"
#include "oneapi/kmeans_init_dense_batch_kernel_ucapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
template class KMeansInitDenseBatchKernelUCAPI<deterministicDense, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<randomDense, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<plusPlusDense, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<parallelPlusDense, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<deterministicCSR, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<randomCSR, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<plusPlusCSR, DAAL_FPTYPE>;
template class KMeansInitDenseBatchKernelUCAPI<parallelPlusCSR, DAAL_FPTYPE>;
} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
