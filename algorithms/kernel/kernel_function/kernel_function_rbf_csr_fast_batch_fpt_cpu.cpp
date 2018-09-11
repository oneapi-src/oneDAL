/* file: kernel_function_rbf_csr_fast_batch_fpt_cpu.cpp */
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

/*
//++
//  Implementation of RBF kernel functions for CSR input data.
//--
*/

#include "kernel_function_rbf_batch_container.h"
#include "kernel_function_rbf_csr_fast_kernel.h"
#include "kernel_function_rbf_csr_fast_impl.i"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace interface1
{

template class BatchContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;

}
namespace internal
{

template class KernelImplRBF<fastCSR, DAAL_FPTYPE, DAAL_CPU>;

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal
