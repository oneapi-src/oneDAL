/* file: minmax_moments.h */
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

services::Status computeMinimumsAndMaximums(low_order_moments::BatchImpl *moments,
                                            daal::data_management::NumericTablePtr &dataTable,
                                            daal::data_management::NumericTablePtr &minimums,
                                            daal::data_management::NumericTablePtr &maximums);

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
