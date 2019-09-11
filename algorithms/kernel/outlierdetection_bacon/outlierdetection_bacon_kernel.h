/* file: outlierdetection_bacon_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
    services::Status compute(NumericTable &data, NumericTable &weights, const Parameter &par);
};

} // namespace internal

} // namespace bacon_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
