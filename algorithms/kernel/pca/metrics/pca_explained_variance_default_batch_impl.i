/* file: pca_explained_variance_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_IMPL_I__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_lapack.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "threading.h"
#include "service_error_handling.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

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

template<Method method, typename algorithmFPType, CpuType cpu>
Status ExplainedVarianceKernel<method, algorithmFPType, cpu>::compute
        (const NumericTable& eigenvalues,
         NumericTable& explainedVariances,
         NumericTable& explainedVariancesRatios,
         NumericTable& noiseVariance)
{
    ReadRows<algorithmFPType, cpu, NumericTable> rowsEigenvalues(const_cast<NumericTable&>(eigenvalues), 0, 1);
    const algorithmFPType* pEigenvalues = rowsEigenvalues.get();

    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsExplainedVariances, &explainedVariances);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsExplainedVariancesRatios, &explainedVariancesRatios);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsNoiseVariance, &noiseVariance);

    algorithmFPType* pExplainedVariances = rowsExplainedVariances.get();
    algorithmFPType* pExplainedVariancesRatios = rowsExplainedVariancesRatios.get();
    algorithmFPType* pNoiseVariance = rowsNoiseVariance.get();

    size_t nFeatures = eigenvalues.getNumberOfColumns();
    size_t nComponents = explainedVariances.getNumberOfColumns();

    algorithmFPType sum = 0;
    algorithmFPType noiseSum = 0;

    for (size_t id = 0; id < nFeatures; ++id)
    {
        sum += pEigenvalues[id];
        if(id >= nComponents)
            noiseSum += pEigenvalues[id];
    }

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t id = 0; id < nComponents; ++id)
    {
        pExplainedVariances[id] = pEigenvalues[id];
        pExplainedVariancesRatios[id] = pEigenvalues[id] / sum;
    }

    int delta = nFeatures - nComponents;
    pNoiseVariance[0] = (delta > 0) ? noiseSum / delta : 0;

    return Status();
}

}
}
}
}
}
}

#endif
