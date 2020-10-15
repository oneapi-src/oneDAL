/* file: svm_train_thunder_batch_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of SVM thunder training algorithm.
//--
*/

#include "src/algorithms/svm/oneapi/svm_train_thunder_kernel_oneapi.h"
#include "src/algorithms/svm/oneapi/svm_train_thunder_oneapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template struct SVMTrainOneAPI<DAAL_FPTYPE, thunder>;

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
