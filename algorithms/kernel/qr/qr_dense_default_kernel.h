/* file: qr_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Declaration of template function that calculate qrs.
//--
*/

#ifndef __QR_FPK_H__
#define __QR_FPK_H__

#include "qr_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{

template<typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRBatchKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    services::Status compute_seq(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    services::Status compute_thr(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

    services::Status compute_pcl(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

};

template<typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QROnlineKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
    services::Status finalizeCompute(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);
};

template<typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRDistributedStep2Kernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r0, NumericTable *r[], const daal::algorithms::Parameter *par = 0, data_management::KeyValueDataCollection *inCollection = NULL);

};

template<typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRDistributedStep3Kernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable *const *a,
                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par = 0);

};

} // namespace daal::internal
}
}
} // namespace daal

#endif
