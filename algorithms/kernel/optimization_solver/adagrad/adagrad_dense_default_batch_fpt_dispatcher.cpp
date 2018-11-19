/* file: adagrad_dense_default_batch_fpt_dispatcher.cpp */
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

//++
//  Implementation of adagrad calculation algorithm container.
//--


#include "adagrad_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::adagrad::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::adagrad::defaultDense)
} // namespace interface1

namespace optimization_solver
{
namespace adagrad
{
namespace interface1
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::adagrad::defaultDense>;

template<>
services::SharedPtr<BatchType> BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}
} // namespace interface1
} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
