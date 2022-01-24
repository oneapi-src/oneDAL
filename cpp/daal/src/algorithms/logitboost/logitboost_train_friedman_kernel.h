/* file: logitboost_train_friedman_kernel.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "src/threading/threading.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"

#include "src/algorithms/logitboost/logitboost_train_kernel.h"

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
template <typename algorithmFPType, CpuType cpu>
struct LogitBoostTrainKernel<friedman, algorithmFPType, cpu> : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const size_t na, NumericTablePtr a[], Model * r, const Parameter * par);
};
} // namespace internal
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
