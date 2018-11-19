/* file: logitboost_predict_dense_default_kernel.h */
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
//  Common functions for Logit Boost predictions calculation
//--
*/

#ifndef __LOGITBOOST_PREDICT_DENSE_DEFAULT_KERNEL_H__
#define __LOGITBOOST_PREDICT_DENSE_DEFAULT_KERNEL_H__

#include "algorithm.h"
#include "service_numeric_table.h"
#include "logitboost_model.h"
#include "daal_defines.h"

#include "logitboost_predict_kernel.h"

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

/**
 *  \brief Specialization of the structure that contains kernels
 *  for Logit Boost prediction calculation using Fast method
 */
template<typename algorithmFPType, CpuType cpu>
struct LogitBoostPredictKernel<defaultDense, algorithmFPType, cpu> : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const NumericTablePtr& a, const Model *m, NumericTable *r, const Parameter *par);
};

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
