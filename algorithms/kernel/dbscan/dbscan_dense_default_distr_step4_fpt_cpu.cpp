/* file: dbscan_dense_default_distr_step4_fpt_cpu.cpp */
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
//  Implementation of DBSCAN functions for distributed computing mode.
//--
*/

#include "dbscan_kernel.h"
#include "dbscan_dense_default_distr_impl.i"
#include "dbscan_container.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{
template class DistributedContainer<step4Local, DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace interface1
namespace internal
{
template class DBSCANDistrStep4Kernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
