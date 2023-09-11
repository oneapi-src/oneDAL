/* file: pca_explained_variance_default_batch_fpt_cpu.cpp */
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
//  Implementation of linear regression single beta quality metrics
//--
*/

#include "src/algorithms/pca/metrics/pca_explained_variance_default_batch_kernel.h"
#include "src/algorithms/pca/metrics/pca_explained_variance_default_batch_impl.i"
#include "src/algorithms/pca/metrics/pca_explained_variance_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
namespace internal
{
template class ExplainedVarianceKernel<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal
