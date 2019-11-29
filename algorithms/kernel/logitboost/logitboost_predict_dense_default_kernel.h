/* file: logitboost_predict_dense_default_kernel.h */
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
template <typename algorithmFPType, CpuType cpu>
struct LogitBoostPredictKernel<defaultDense, algorithmFPType, cpu> : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const NumericTablePtr & a, const Model * m, NumericTable * r, const Parameter * par);
};
} // namespace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
