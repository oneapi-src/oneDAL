/* file: adagrad_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of adagrad calculation.
//--

#include "src/algorithms/optimization_solver/adagrad/adagrad_batch_container.h"
#include "src/algorithms/optimization_solver/adagrad/adagrad_dense_default_kernel.h"
#include "src/algorithms/optimization_solver/adagrad/adagrad_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
namespace internal
{
template class AdagradKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
