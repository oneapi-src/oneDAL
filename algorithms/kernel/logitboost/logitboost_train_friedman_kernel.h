/* file: logitboost_train_friedman_kernel.h */
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
//  Common functions for Logit Boost model training
//--
*/
/*
//
//  REFERENCES
//
//  1. J. Friedman, T. Hastie, R. Tibshirani.
//     Additive logistic regression: a statistical view of boosting,
//     The annals of Statistics, 2000, v28 N2, pp. 337-407
//  2. J. Friedman, T. Hastie, R. Tibshirani.
//     The Elements of Statistical Learning:
//     Data Mining, Inference, and Prediction,
//     Springer, 2001.
//
*/

#ifndef __LOGITBOOST_TRAIN_FRIEDMAN_KERNEL_H__
#define __LOGITBOOST_TRAIN_FRIEDMAN_KERNEL_H__

#include "threading.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"

#include "logitboost_train_kernel.h"

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
 *  \brief Specialization of the structure that contains kernels for
 *  Logit Boost model training using Friedman method
 */
template<typename algorithmFPType, CpuType cpu>
struct LogitBoostTrainKernel<friedman, algorithmFPType, cpu> : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const size_t na, NumericTablePtr a[], Model *r, const Parameter *par);
};

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
