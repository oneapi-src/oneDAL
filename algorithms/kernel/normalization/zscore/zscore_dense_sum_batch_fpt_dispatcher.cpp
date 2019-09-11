/* file: zscore_dense_sum_batch_fpt_dispatcher.cpp */
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
//  Implementation of zscore normalization algorithm container.
//
//--


#include "zscore_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(normalization::zscore::interface3::BatchContainer, batch, DAAL_FPTYPE, normalization::zscore::sumDense)
} // namespace algorithms
} // namespace daal


namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface3
{
template<typename algorithmFPType, daal::algorithms::normalization::zscore::Method method>
Batch<algorithmFPType, method>::Batch()
{
    _par = new ParameterType();
    initialize();
}

template<typename algorithmFPType, daal::algorithms::normalization::zscore::Method method>
Batch<algorithmFPType, method>::Batch(const Batch &other): BatchImpl(other)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
template Batch<DAAL_FPTYPE, normalization::zscore::sumDense>::Batch();
template Batch<DAAL_FPTYPE, normalization::zscore::defaultDense>::Batch();
template Batch<DAAL_FPTYPE, normalization::zscore::sumDense>::Batch(const Batch &);
template Batch<DAAL_FPTYPE, normalization::zscore::defaultDense>::Batch(const Batch &);
} // namespace interface3
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
