/* file: zscore_v2.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "src/algorithms/normalization/zscore/inner/zscore_types_v2.h"
#include "src/algorithms/normalization/zscore/inner/zscore_result_v1.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface2
{
BaseParameter::BaseParameter() : resultsToCompute(none) {}

} // namespace interface2

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
