/* file: cosdistance_kernel.h */
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
//  Declaration of kernel class for computation of cosine distance.
//--
*/

#ifndef __COSDISTANCE_KERNEL_H__
#define __COSDISTANCE_KERNEL_H__

#include "cosine_distance.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class DistanceKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable *const *a, const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal

#endif
