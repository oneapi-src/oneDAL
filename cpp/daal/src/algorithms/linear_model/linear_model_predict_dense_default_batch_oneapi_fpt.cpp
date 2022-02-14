/* file: linear_model_predict_dense_default_batch_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of prediction stage of linear regression algorithm.
//--
*/

#include "src/algorithms/linear_model/oneapi/linear_model_predict_kernel_oneapi.h"
#include "src/algorithms/linear_model/oneapi/linear_model_predict_dense_default_batch_oneapi_impl.i"

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
template class DAAL_EXPORT PredictKernelOneAPI<DAAL_FPTYPE, defaultDense>;
} // namespace internal
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal
