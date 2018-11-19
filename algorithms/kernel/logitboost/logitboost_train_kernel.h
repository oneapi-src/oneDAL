/* file: logitboost_train_kernel.h */
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
//  Declaration of structure containing kernels for logit boost model
//  training.
//--
*/

#ifndef __LOGITBOOST_TRAIN_KERNEL_H__
#define __LOGITBOOST_TRAIN_KERNEL_H__

#include "logitboost_model.h"
#include "logitboost_training_types.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{

/**
 *  \brief Construct Logit Boost classifier model.
 *
 *  \param a[in]    Array of numeric tables contating input data
 *                  a[0] holds input matrix of features X
 *                  a[1] holds input matrix of class labels Y
 *  \param r[out]   Resulting model
 *  \param par[in]  Logit Boost algorithm parameters
 */
template <Method method, typename algorithmFPType, CpuType cpu>
struct LogitBoostTrainKernel : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const size_t na, NumericTablePtr a[], Model *r, const Parameter *par);
};

} // namespace daal::algorithms::logitboost::training::internal
}
}
}
} // namespace daal

#endif
