/* file: outlierdetection_bacon_kernel.h */
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
//  Declaration of template structs for multivariate outlier detection
//--
*/

#ifndef __OUTLIERDETECTION_BACON_KERNEL_H__
#define __OUTLIERDETECTION_BACON_KERNEL_H__

#include "outlier_detection_bacon.h"
#include "kernel.h"
#include "service_numeric_table.h"
#include "service_math.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace bacon_outlier_detection
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
struct OutlierDetectionKernel : public Kernel
{
    services::Status compute(NumericTable & data, NumericTable & weights, const Parameter & par);
};

} // namespace internal

} // namespace bacon_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
