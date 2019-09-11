/* file: logitboost_predict_kernel.h */
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
//  Declaration of template function that computes Logit Boost
//  classification results.
//--
*/

#ifndef __LOGITBOOST_PREDICT_KERNEL_H__
#define __LOGITBOOST_PREDICT_KERNEL_H__

#include "logitboost_predict.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
struct LogitBoostPredictKernel : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    /**
     *  \brief Calculate Logit Boost classification results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Logit Boost model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param par[in]  Logit Boost algorithm parameters
     */
    services::Status compute( NumericTablePtr a, const Model *m, NumericTable *r, const Parameter *par );
};

} // namespace daal::algorithms::logitboost::prediction::internal
}
}
}
} // namespace daal

#endif
