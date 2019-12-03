/* file: minmax_moments.cpp */
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

#include "minmax_moments.h"

using namespace daal::services;
using namespace daal::data_management;

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
Status computeMinimumsAndMaximums(low_order_moments::BatchImpl * moments, NumericTablePtr & dataTable, NumericTablePtr & minimums,
                                  NumericTablePtr & maximums)
{
    minimums = dataTable->basicStatistics.get(NumericTableIface::minimum);
    maximums = dataTable->basicStatistics.get(NumericTableIface::maximum);

    if (!minimums || !maximums)
    {
        moments->parameter.estimatesToCompute = low_order_moments::estimatesMinMax;
        moments->input.set(low_order_moments::data, dataTable);
        DAAL_CHECK(moments->computeNoThrow(), ErrorMinAndMaxComputing);

        minimums = moments->getResult()->get(low_order_moments::minimum);
        maximums = moments->getResult()->get(low_order_moments::maximum);
    }

    return Status();
}

} // namespace internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
