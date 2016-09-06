/* file: xavier_initializer_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of Xavier calculation functions.
//--


#include "xavier_initializer_batch_container.h"
#include "xavier_initializer_kernel.h"
#include "xavier_initializer_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{

namespace interface1
{
template class neural_networks::initializers::xavier::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1

namespace internal
{
template class XavierKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // internal

}
}
}
}
}
