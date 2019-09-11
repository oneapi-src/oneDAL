/* file: cordistance_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of correlation distance calculation functions.
//--
*/


#include "cordistance_batch_container.h"
#include "cordistance_kernel.h"
#include "cordistance_batch_impl.i"

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace interface1
{

template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

}
namespace internal
{

template class DistanceKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
