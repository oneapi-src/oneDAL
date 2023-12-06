/* file: pca_dense_base.h */
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
//  Declaration of template structs that calculate PCA Correlation.
//--
*/

#ifndef __PCA_DENSE_BASE_H__
#define __PCA_DENSE_BASE_H__

#include "src/algorithms/kernel.h"
#include "src/algorithms/pca/pca_dense_base_iface.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"
namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::internal;
template <typename algorithmFPType, CpuType cpu>
class PCADenseBase : public Kernel
{
public:
    services::Status signFlipEigenvectors(NumericTable & eigenvectors) const;
    services::Status fillTable(NumericTable & table, algorithmFPType val) const;
    services::Status copyTable(NumericTable & source, NumericTable & dest) const;
    services::Status computeExplainedVariancesRatio(const data_management::NumericTable & eigenvalues,
                                                    const data_management::NumericTable & variances,
                                                    data_management::NumericTable & explained_variances_ratio);

private:
    void signFlipArray(size_t size, algorithmFPType * source) const;
};
template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::computeExplainedVariancesRatio(const data_management::NumericTable & eigenvalues,
                                                                                    const data_management::NumericTable & variances,
                                                                                    data_management::NumericTable & explained_variances_ratio)
{
    const size_t nComponents = eigenvalues.getNumberOfColumns();
    const size_t nColumns    = variances.getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> eigenValuesBlock(const_cast<data_management::NumericTable &>(eigenvalues), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(eigenValuesBlock);
    const algorithmFPType * const eigenValuesArray = eigenValuesBlock.get();
    ReadRows<algorithmFPType, cpu> variancesBlock(const_cast<data_management::NumericTable &>(variances), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(variancesBlock);
    const algorithmFPType * const variancesBlockArray = variancesBlock.get();
    WriteRows<algorithmFPType, cpu> explainedVariancesRatioBlock(explained_variances_ratio, 0, 1);
    DAAL_CHECK_MALLOC(explainedVariancesRatioBlock.get());
    algorithmFPType * explainedVariancesRatioArray = explainedVariancesRatioBlock.get();
    algorithmFPType sum                            = 0;
    for (size_t i = 0; i < nColumns; i++)
    {
        sum += variancesBlockArray[i];
    }
    if (sum <= 0) return services::Status(services::ErrorIncorrectEigenValuesSum);
    for (size_t i = 0; i < nComponents; i++)
    {
        explainedVariancesRatioArray[i] = eigenValuesArray[i] / sum;
    }
    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif // __PCA_DENSE_BASE_H__
