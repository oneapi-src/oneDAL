/* file: sgd_dense_momentum_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//++
//  Implementation of sgd calculation algorithm container.
//--

#include "src/algorithms/optimization_solver/sgd/sgd_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(optimization_solver::sgd::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::sgd::momentum)

namespace optimization_solver
{
namespace sgd
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::sgd::momentum>;

template <>
services::SharedPtr<BatchType> BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
