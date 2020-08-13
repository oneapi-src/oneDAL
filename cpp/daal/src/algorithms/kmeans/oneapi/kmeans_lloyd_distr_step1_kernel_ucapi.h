/* file: kmeans_lloyd_distr_step1_kernel_ucapi.h */
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
//  Implementation of K-means Distr Step1 Kernel for GPU.
//--
*/

#ifndef __KMEANS_LLOYD_DISTR_STEP1_KERNEL_UCAPI_H__
#define __KMEANS_LLOYD_DISTR_STEP1_KERNEL_UCAPI_H__

#include "src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_kernel_base_ucapi.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <typename algorithmFPType>
class KMeansDistributedStep1KernelUCAPI : public KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>
{
public:
    services::Status compute(size_t na, const NumericTable * const * a, size_t nr, const NumericTable * const * r, const Parameter * par);
    services::Status finalizeCompute(size_t na, const NumericTable * const * a, size_t nr, const NumericTable * const * r, const Parameter * par);
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
