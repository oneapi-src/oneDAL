/* file: kernel_function_linear_csr_fast_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/*
//++
//  Implementation of linear kernel functions for CSR input data.
//--
*/

#include "kernel_function_linear_batch_container.h"
#include "kernel_function_linear_csr_fast_kernel.h"
#include "kernel_function_linear_csr_fast_impl.i"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;

}
namespace internal
{
template class KernelImplLinear<fastCSR, DAAL_FPTYPE, DAAL_CPU>;

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal
