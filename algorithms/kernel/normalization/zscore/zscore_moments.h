/* file: zscore_moments.h */
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

#ifndef __ZSCORE_MOMENTS_H__
#define __ZSCORE_MOMENTS_H__

#include "zscore.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{
template<typename algorithmFPType>
services::Status computeMeansAndVariances(low_order_moments::BatchImpl *moments,
                                          const daal::data_management::NumericTable &dataTable,
                                          algorithmFPType* resultMean,
                                          algorithmFPType* resultVariance);

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
