/* file: pca_transform_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of pca transformation algorithm.
//--
*/

#include "pca_transform_container.h"
#include "pca_transform_kernel.h"
#include "pca_transform_dense_default_batch_impl.i"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, transform::defaultDense, DAAL_CPU>;
}
namespace internal
{
template class TransformKernel<DAAL_FPTYPE, transform::defaultDense, DAAL_CPU>;
}
}
}
}
}
