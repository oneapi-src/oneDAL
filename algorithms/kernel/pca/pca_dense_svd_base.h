/* file: pca_dense_svd_base.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_BASE_H__
#define __PCA_DENSE_SVD_BASE_H__

#include "service_numeric_table.h"
#include "pca_dense_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
enum InputDataType
{
    nonNormalizedDataset = 0, /*!< Original, non-normalized data set */
    normalizedDataset    = 1, /*!< Normalized data set whose feature vectors have zero average and unit variance */
    correlation          = 2  /*!< Correlation matrix */
};

template <typename algorithmFPType, CpuType cpu>
class PCASVDKernelBase : public PCADenseBase<algorithmFPType, cpu>
{
public:
    PCASVDKernelBase() {}
    virtual ~PCASVDKernelBase() {}

protected:
    services::Status scaleSingularValues(data_management::NumericTable & eigenvaluesTable, size_t nVectors);
};

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDKernelBase<algorithmFPType, cpu>::scaleSingularValues(NumericTable & eigenvaluesTable, size_t nVectors)
{
    const size_t nFeatures = eigenvaluesTable.getNumberOfColumns();
    daal::internal::WriteRows<algorithmFPType, cpu> block(eigenvaluesTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(block);
    algorithmFPType * eigenvalues = block.get();

    for (size_t i = 0; i < nFeatures; i++)
    {
        eigenvalues[i] = eigenvalues[i] * eigenvalues[i] / (nVectors - 1);
    }
    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
