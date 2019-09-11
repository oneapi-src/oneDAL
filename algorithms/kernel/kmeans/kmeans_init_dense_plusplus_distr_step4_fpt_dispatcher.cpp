/* file: kmeans_init_dense_plusplus_distr_step4_fpt_dispatcher.cpp */
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
//  Implementation of k-means plus plus initialization method for K-means algorithm
//--
*/

#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step4Local,  DAAL_FPTYPE, kmeans::init::plusPlusDense)

namespace kmeans
{
namespace init
{
namespace interface2
{

using DistributedType = Distributed<step4Local, DAAL_FPTYPE, kmeans::init::plusPlusDense>;

template <>
DistributedType::Distributed(size_t nClusters) : DistributedBase(new ParameterType(nClusters)),
    parameter(*static_cast<ParameterType*>(_par))
{
    initialize();
}

template <>
DistributedType::Distributed(const DistributedType &other) : DistributedBase(new ParameterType(other.parameter)),
    parameter(*static_cast<ParameterType*>(_par)), input(other.input)
{
    initialize();
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace daal::algorithms
} // namespace daal
