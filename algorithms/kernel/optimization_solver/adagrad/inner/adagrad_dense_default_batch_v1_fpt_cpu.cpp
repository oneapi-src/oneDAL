/* file: adagrad_dense_default_batch_v1_fpt_cpu.cpp */
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

//++
//  Implementation of adagrad calculation.
//--


#include "adagrad_batch_container_v1.h"
#include "adagrad_dense_default_kernel_v1.h"
#include "adagrad_dense_default_v1_impl.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{

namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

namespace internal
{
template class I1AdagradKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
