/* file: kmeans_init_csr_plusplus_distr_step2_fpt_dispatcher_v1.cpp */
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
//  Implementation of K-means plus plus initialization method for K-means algorithm
//--
*/

#include "kmeans/inner/kmeans_init_container_v1.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_KM(kmeans::init::interface1::DistributedContainer, distributed, step2Local,  DAAL_FPTYPE, kmeans::init::plusPlusCSR)
} // namespace daal::algorithms
} // namespace daal
