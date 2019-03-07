/* file: kernel_function_linear_csr_fast_batch_fpt_dispatcher.cpp */
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
//  Implementation of linear kernel function container for CSR input data.
//--
*/

#include "kernel_function_linear.h"
#include "kernel_function_linear_batch_container.h"
#include "kernel_function_linear_csr_fast_kernel.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kernel_function::linear::BatchContainer, batch, DAAL_FPTYPE, kernel_function::linear::fastCSR)
}
} // namespace algorithms
} // namespace daal
