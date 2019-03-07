/* file: kmeans_init_csr_random_distr_step1_fpt_dispatcher.cpp */
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
//  Implementation of K-means initialization random algorithm container for CSR
//--
*/

#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step1Local,  DAAL_FPTYPE, kmeans::init::randomCSR)
}

namespace kmeans
{
namespace init
{
namespace interface2
{

using DistributedType = Distributed<step1Local, DAAL_FPTYPE, kmeans::init::randomCSR>;

template <>
DistributedType::Distributed(size_t nClusters, size_t nRowsTotal, size_t offset) :
    DistributedBase(new ParameterType(nClusters, offset)), parameter(*static_cast<ParameterType*>(_par))
{
    initialize();
    parameter.nRowsTotal = nRowsTotal;
}

template <>
DistributedType::Distributed(const DistributedType &other) :
    DistributedBase(new ParameterType(other.parameter)), parameter(*static_cast<ParameterType*>(_par))
{
    initialize();
    input.set(data, other.input.get(data));
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace daal::algorithms
} // namespace daal
