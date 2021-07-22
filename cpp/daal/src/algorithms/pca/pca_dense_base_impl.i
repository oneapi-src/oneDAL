/* file: pca_dense_base_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "src/algorithms/pca/pca_dense_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::copyTable(NumericTable & source, NumericTable & dest) const
{
    size_t nElements = dest.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> sourceBlock(source, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(sourceBlock);
    WriteOnlyRows<algorithmFPType, cpu> destBlock(dest, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(destBlock);
    const algorithmFPType * sourceData = sourceBlock.get();
    algorithmFPType * destData         = destBlock.get();
    for (size_t id = 0; id < nElements; ++id)
    {
        destData[id] = sourceData[id];
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::signFlipEigenvectors(NumericTable & eigenvectors) const
{
    size_t nFeatures = eigenvectors.getNumberOfColumns();
    size_t nVectors  = eigenvectors.getNumberOfRows();
    WriteRows<algorithmFPType, cpu> eigenvectorsBlock(eigenvectors, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(eigenvectorsBlock);
    algorithmFPType * eigenvectorsData = eigenvectorsBlock.get();
    for (size_t id = 0; id < nVectors; ++id)
    {
        signFlipArray(nFeatures, eigenvectorsData + id * nFeatures);
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::fillTable(NumericTable & table, algorithmFPType val) const
{
    size_t nElements = table.getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> tableBlock(table, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(tableBlock);
    algorithmFPType * tableData = tableBlock.get();
    for (size_t id = 0; id < nElements; ++id)
    {
        tableData[id] = val;
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
void PCADenseBase<algorithmFPType, cpu>::signFlipArray(size_t size, algorithmFPType * source) const
{
#define FABS(X) ((X) > (algorithmFPType)0 ? (X) : -(X))
    algorithmFPType smax = source[0];
    algorithmFPType max  = FABS(smax);
    for (size_t id = 1; id < size; ++id)
    {
        algorithmFPType tmp = FABS(source[id]);
        if (tmp > max)
        {
            max  = tmp;
            smax = source[id];
        }
    }
    if (smax < 0)
    {
        for (size_t id = 0; id < size; ++id)
        {
            source[id] = -source[id];
        }
    }
#undef FABS
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
