/* file: dbscan_dense_default_distr_step10_fpt_dispatcher.cpp */
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
//  Implementation of DBSCAN algorithm container for distributed
//  computing mode.
//--
*/

#include "dbscan_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(dbscan::DistributedContainer, distributed, step10Local,  \
    DAAL_FPTYPE, dbscan::defaultDense)

namespace dbscan
{
namespace interface1
{

using DistributedType = Distributed<step10Local, DAAL_FPTYPE, defaultDense>;

template <>
DistributedType::Distributed(size_t blockIndex, size_t nBlocks)
{
    ParameterType *par = new ParameterType();
    par->blockIndex = blockIndex;
    par->nBlocks = nBlocks;

    _par = par;
    initialize();
}

template <>
DistributedType::Distributed(const DistributedType &other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
