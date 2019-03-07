/* file: low_order_moments_csr_fast_distr_step2_fpt_cpu.cpp */
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
//  Implementation of low order moments kernel.
//--
*/

#include "low_order_moments_container.h"
#include "low_order_moments_distributed_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace interface1
{
template class DistributedContainer<step2Master, DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
}
}
}
