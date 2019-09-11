/* file: sgd_dense_momentum_batch_v1_fpt_dispatcher.cpp */
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
//  Implementation of sgd calculation algorithm container.
//--


#include "sgd_batch_container_v1.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::sgd::interface1::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::sgd::momentum)

namespace optimization_solver
{
namespace sgd
{
namespace interface1
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::sgd::momentum>;

template<>
services::SharedPtr<BatchType> BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}

} // namespace interface1

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
