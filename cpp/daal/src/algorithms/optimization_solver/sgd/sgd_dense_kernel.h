/* file: sgd_dense_kernel.h */
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
//  Declaration of template function that calculate sgd.
//--

#ifndef __SGD_DENSE_KERNEL_H__
#define __SGD_DENSE_KERNEL_H__

#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "algorithms/optimization_solver/sgd/sgd_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class SGDKernel : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{};

} // namespace internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
