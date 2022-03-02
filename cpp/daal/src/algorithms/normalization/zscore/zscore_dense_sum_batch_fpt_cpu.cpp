/* file: zscore_dense_sum_batch_fpt_cpu.cpp */
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

//++
//  Implementation of zscore normalization calculation functions.
//
//--

#include "src/algorithms/normalization/zscore/zscore_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface3
{
template class BatchContainer<DAAL_FPTYPE, sumDense, DAAL_CPU>;
} // namespace interface3

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
