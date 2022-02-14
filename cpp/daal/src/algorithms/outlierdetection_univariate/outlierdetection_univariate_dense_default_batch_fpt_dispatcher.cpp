/* file: outlierdetection_univariate_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of container for univariate outlier detection.
//--
*/

#include "algorithms/outlier_detection/outlier_detection_univariate.h"
#include "src/algorithms/outlierdetection_univariate/outlierdetection_univariate_batch_container.h"
#include "src/algorithms/outlierdetection_univariate/outlierdetection_univariate_kernel.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(univariate_outlier_detection::BatchContainer, batch, DAAL_FPTYPE, univariate_outlier_detection::defaultDense)
} // namespace algorithms

} // namespace daal
