/* file: svd_dense_default_online_fpt_cpu.cpp */
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

/*
//++
//  Instantiation of SVD algorithm classes.
//--
*/

#include "src/algorithms/svd/svd_dense_default_kernel.h"
#include "src/algorithms/svd/svd_dense_default_online_impl.i"
#include "src/algorithms/svd/svd_dense_default_container.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{
template class OnlineContainer<DAAL_FPTYPE, daal::algorithms::svd::defaultDense, DAAL_CPU>;
}
namespace internal
{
template class SVDOnlineKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
} // namespace svd
} // namespace algorithms
} // namespace daal
