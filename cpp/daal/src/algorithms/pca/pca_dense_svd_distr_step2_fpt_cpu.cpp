/* file: pca_dense_svd_distr_step2_fpt_cpu.cpp */
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
//  Implementation of PCA calculation functions.
//--

#include "src/algorithms/pca/pca_dense_svd_distr_step2_container.h"
#include "src/algorithms/pca/pca_dense_svd_distr_step2_kernel.h"
#include "src/algorithms/pca/pca_dense_svd_distr_step2_impl.i"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
template class DistributedContainer<step2Master, DAAL_FPTYPE, svdDense, DAAL_CPU>;
}
namespace internal
{
template class PCASVDStep2MasterKernel<DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace pca
} // namespace algorithms
} // namespace daal
