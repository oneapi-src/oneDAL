/* file: tanh_csr_fast_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of hyperbolic tangent calculation functions for AVX2.
//--

#include "tanh_batch_container.h"
#include "tanh_base.h"
#include "tanh_csr_fast_kernel.h"
#include "tanh_impl.i"
#include "tanh_csr_fast_impl.i"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace tanh
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class TanhKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
template class TanhKernelBase<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
}
}
}
}
