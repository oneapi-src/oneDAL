/* file: kmeans_init_csr_deterministic_distr_step2_fpt_dispatcher.cpp */
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

#include "kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, kmeans::init::deterministicCSR)

namespace kmeans
{
namespace init
{
namespace interface2
{

using DistributedType = Distributed<step2Master, DAAL_FPTYPE, kmeans::init::deterministicCSR>;

template <>
DistributedType::Distributed(size_t nClusters, size_t offset) : DistributedBase(new ParameterType(nClusters, offset)),
    parameter(*static_cast<ParameterType*>(_par))
{
    Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master,
        DAAL_FPTYPE, kmeans::init::deterministicCSR)(&_env);
    _in  = &input;
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace daal::algorithms
} // namespace daal
