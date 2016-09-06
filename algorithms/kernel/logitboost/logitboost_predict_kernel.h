/* file: logitboost_predict_kernel.h */
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
//  Declaration of template function that computes Logit Boost
//  classification results.
//--
*/

#ifndef __LOGITBOOST_PREDICT_KERNEL_H__
#define __LOGITBOOST_PREDICT_KERNEL_H__

#include "logitboost_predict.h"
#include "kernel.h"
#include "numeric_table.h"

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
    /**
     *  \brief Calculate Logit Boost classification results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Logit Boost model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param par[in]  Logit Boost algorithm parameters
     */
    void compute( NumericTablePtr a, const Model *m, NumericTable *r, const Parameter *par );
};

} // namespace daal::algorithms::logitboost::prediction::internal
}
}
}
} // namespace daal

#endif
