/* file: linear_model_predict_kernel_oneapi.h */
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
//  Declaration of template function that computes linear regression
//  prediction results.
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_KERNEL_ONEAPI_H__
#define __LINEAR_MODEL_PREDICT_KERNEL_ONEAPI_H__

#include "linear_model_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{
template <typename algorithmFpType, prediction::Method method>
class PredictKernelOneAPI : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute linear regression prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Linear regression model obtained on training stage
     *  \param r[out]   Prediction results
     */
    services::Status compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r);
};

template <typename algorithmFPType>
class PredictKernelOneAPI<algorithmFPType, defaultDense> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r);

protected:
    services::Status addBetaIntercept(const services::Buffer<algorithmFPType> & betaTable, const size_t nBetas,
                                      services::Buffer<algorithmFPType> & yTable, const size_t yNRows, const size_t yNCols);
};

} // namespace internal
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
