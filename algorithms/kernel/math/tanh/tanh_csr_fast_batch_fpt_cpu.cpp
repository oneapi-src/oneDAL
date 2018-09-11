/* file: tanh_csr_fast_batch_fpt_cpu.cpp */
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

//++
//  Implementation of hyperbolic tangent calculation functions.
//--

#include "tanh_batch_container.h"
#include "tanh_base.h"
#include "tanh_csr_fast_kernel.h"
#include "tanh_impl.i"
#include "tanh_csr_fast_impl.i"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace tanh
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class TanhKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
template class TanhKernelBase<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
}
}
}
}
