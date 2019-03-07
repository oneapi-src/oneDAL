/* file: truncated_gaussian_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of truncated gaussian initializer functions.
//--


#include "truncated_gaussian_batch_container.h"
#include "truncated_gaussian_kernel.h"
#include "truncated_gaussian_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{

namespace interface1
{
template class neural_networks::initializers::truncated_gaussian::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1

namespace internal
{
template class TruncatedGaussianKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // internal

}
}
}
}
}
