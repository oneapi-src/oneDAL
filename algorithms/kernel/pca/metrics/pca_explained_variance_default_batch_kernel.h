/* file: pca_explained_variance_default_batch_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_KERNEL_H__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_KERNEL_H__

#include "pca_explained_variance_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "algorithm_base_common.h"


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
    NumericTable* rms;
    NumericTable* variance;
    NumericTable** betaCovariances;
    NumericTable* zScore;
    NumericTable* confidenceIntervals;
    NumericTable* inverseOfXtX;

    ExplainedVarianceOutput(size_t nResponses);
    ~ExplainedVarianceOutput();
};


template<Method method, typename algorithmFPType, CpuType cpu>
class ExplainedVarianceKernel : public daal::algorithms::Kernel
{
public:
    virtual ~ExplainedVarianceKernel() {}

    services::Status compute(const NumericTable& eigenvalues,
                             NumericTable& explainedVariances,
                             NumericTable& explainedVariancesRatios,
                             NumericTable& noiseVariance);
};

}
}
}
}
}
}

#endif
