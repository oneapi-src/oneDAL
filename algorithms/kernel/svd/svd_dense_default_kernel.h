/* file: svd_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{

template<typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDBatchKernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    Status compute_seq(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const Parameter *par = 0);

    Status compute_thr(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const Parameter *par = 0);

    Status compute_pcl(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const Parameter *par = 0);

};

template<typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDOnlineKernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
    Status finalizeCompute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep2Kernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep3Kernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable *const *a,
                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

};

} // namespace daal::internal
}
}
} // namespace daal

#endif
