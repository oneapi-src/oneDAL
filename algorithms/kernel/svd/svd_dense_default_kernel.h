/* file: svd_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Declaration of template function that calculate svds.
//--
*/

#ifndef __SVD_FPK_H__
#define __SVD_FPK_H__

#include "svd_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{

template<typename interm, daal::algorithms::svd::Method method, CpuType cpu>
class SVDBatchKernel : public Kernel
{
public:
    void compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    void compute_seq(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    void compute_thr(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename interm, daal::algorithms::svd::Method method, CpuType cpu>
class SVDOnlineKernel : public Kernel
{
public:
    void compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
    void finalizeCompute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename interm, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep2Kernel : public Kernel
{
public:
    void compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename interm, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep3Kernel : public Kernel
{
public:
    void compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

} // namespace daal::internal
}
}
} // namespace daal

#endif
