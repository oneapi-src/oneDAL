/* file: gbt_regression_init_distr_step2_fpt_cpu.cpp */
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
//  Implementation of  container for initializing gradient boosted trees
//  regression training algorithm in the distributed processing mode
//--
*/

#include "gbt_regression_init_kernel.h"
#include "gbt_regression_init_impl.i"
#include "gbt_regression_init_container.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace interface1
{
template class DistributedContainer<step2Master, DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace interface1
namespace internal
{
template class RegressionInitStep2MasterKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
