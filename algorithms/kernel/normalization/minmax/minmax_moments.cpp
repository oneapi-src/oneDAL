/* file: minmax_moments.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

Status computeMinimumsAndMaximums(low_order_moments::BatchImpl *moments, NumericTablePtr &dataTable,
                                  NumericTablePtr &minimums, NumericTablePtr &maximums)
{
    minimums = dataTable->basicStatistics.get(NumericTableIface::minimum);
    maximums = dataTable->basicStatistics.get(NumericTableIface::maximum);

    if(!minimums || !maximums)
    {
        moments->parameter.estimatesToCompute = low_order_moments::estimatesMinMax;
        moments->input.set(low_order_moments::data, dataTable);
        DAAL_CHECK(moments->computeNoThrow(), ErrorMinAndMaxComputing);

        minimums = moments->getResult()->get(low_order_moments::minimum);
        maximums = moments->getResult()->get(low_order_moments::maximum);
   }

   return Status();
}

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
