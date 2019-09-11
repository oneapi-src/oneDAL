/* file: linear_regression_train_dense_qr_impl.i */
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
//  Implementation of auxiliary functions for linear regression qrDense method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__

#include "linear_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::qr::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::qrDense, cpu>::compute(const NumericTable &x,
                                                                     const NumericTable &y,
                                                                     NumericTable &r, NumericTable &qty,
                                                                     NumericTable &beta, bool interceptFlag) const
{
    Status st = UpdateKernelType::compute(x, y, r, qty, true, interceptFlag);
    if (st)
     st = FinalizeKernelType::compute(r, qty, r, qty, beta, interceptFlag);
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::qrDense, cpu>::compute(const NumericTable &x,
                                                                      const NumericTable &y,
                                                                      NumericTable &r, NumericTable &qty,
                                                                      bool interceptFlag) const
{
    return UpdateKernelType::compute(x, y, r, qty, false, interceptFlag);
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::qrDense, cpu>::finalizeCompute(const NumericTable &r,
                                                                              const NumericTable &qty,
                                                                              NumericTable &rFinal,
                                                                              NumericTable &qtyFinal,
                                                                              NumericTable &beta, bool interceptFlag) const
{
    return FinalizeKernelType::compute(r, qty, rFinal, qtyFinal, beta, interceptFlag);
}

} // internal
} // training
} // linear_regression
} // algorithms
} // daal

#endif
