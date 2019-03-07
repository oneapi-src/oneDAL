/* file: kmeans_init_csr_deterministic_batch_fpt_cpu.cpp */
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
//  Implementation of K-means initialization algorithm container for CSR
//--
*/

#include "kmeans_init_kernel.h"
#include "kmeans_init_impl.i"
#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface2
{
template class BatchContainer<DAAL_FPTYPE, deterministicCSR, DAAL_CPU>;
}
namespace internal
{
template class KMeansInitKernel<deterministicCSR, DAAL_FPTYPE, DAAL_CPU>;
} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
