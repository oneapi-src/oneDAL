/* file: dbscan_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of DBSCAN container.
//--
*/

#include "dbscan_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(dbscan::BatchContainer, batch, DAAL_FPTYPE, dbscan::defaultDense)

namespace dbscan
{
namespace interface1
{

template <>
Batch<DAAL_FPTYPE, dbscan::defaultDense>::Batch(DAAL_FPTYPE epsilon, size_t minObservations)
{
    _par = new ParameterType(epsilon, minObservations);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, dbscan::defaultDense>;
template <>
Batch<DAAL_FPTYPE, dbscan::defaultDense>::Batch(const BatchType &other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
