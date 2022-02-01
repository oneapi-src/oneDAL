/* file: pca_explained_variance_default_batch_kernel.h */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_KERNEL_H__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_KERNEL_H__

#include "algorithms/pca/pca_explained_variance_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"

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
namespace internal
{
using namespace daal::data_management;
struct ExplainedVarianceOutput
{
    NumericTable * rms;
    NumericTable * variance;
    NumericTable ** betaCovariances;
    NumericTable * zScore;
    NumericTable * confidenceIntervals;
    NumericTable * inverseOfXtX;

    ExplainedVarianceOutput(size_t nResponses);
    ~ExplainedVarianceOutput();
};

template <Method method, typename algorithmFPType, CpuType cpu>
class ExplainedVarianceKernel : public daal::algorithms::Kernel
{
public:
    virtual ~ExplainedVarianceKernel() {}

    services::Status compute(const NumericTable & eigenvalues, NumericTable & explainedVariances, NumericTable & explainedVariancesRatios,
                             NumericTable & noiseVariance);
};

} // namespace internal
} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
