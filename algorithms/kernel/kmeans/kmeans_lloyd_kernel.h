/* file: kmeans_lloyd_kernel.h */
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef _KMEANS_FPK_H
#define _KMEANS_FPK_H

#include "kmeans_types.h"
//#include "kmeans_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansBatchKernel : public Kernel
{
public:
    services::Status compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansDistributedStep1Kernel : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable * const * a, size_t nr, const NumericTable * const * r, const Parameter * par);
    services::Status finalizeCompute(size_t na, const NumericTable * const * a, size_t nr, const NumericTable * const * r, const Parameter * par);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansDistributedStep2Kernel : public Kernel
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
