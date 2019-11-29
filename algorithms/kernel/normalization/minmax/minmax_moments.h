/* file: minmax_moments.h */
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

#ifndef __MINMAX_MOMENTS_H__
#define __MINMAX_MOMENTS_H__

#include "normalization/minmax.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace internal
{
services::Status computeMinimumsAndMaximums(low_order_moments::BatchImpl * moments, daal::data_management::NumericTablePtr & dataTable,
                                            daal::data_management::NumericTablePtr & minimums, daal::data_management::NumericTablePtr & maximums);

} // namespace internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
