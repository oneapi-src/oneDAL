/* file: pca_dense_base.h */
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
//  Declaration of template structs that calculate PCA Correlation.
//--
*/

#ifndef __PCA_DENSE_BASE_H__
#define __PCA_DENSE_BASE_H__

#include "service_defines.h"
#include "service_numeric_table.h"
#include "services/error_handling.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{



template <typename algorithmFPType, CpuType cpu>
class PCADenseBase: public Kernel
{
public:
    services::Status signFlipEigenvectors(NumericTable& eigenvectors) const;
    services::Status fillTable(NumericTable& table, algorithmFPType val) const;
    services::Status copyTable(NumericTable& source, NumericTable& dest) const;

private:
    void signFlipArray(size_t size, algorithmFPType *source) const;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::copyTable(NumericTable& source, NumericTable& dest) const
{
    size_t nElements = dest.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> sourceBlock(source, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(sourceBlock);
    WriteOnlyRows<algorithmFPType, cpu> destBlock(dest, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(destBlock);
    const algorithmFPType *sourceData = sourceBlock.get();
    algorithmFPType *destData = destBlock.get();
    for (size_t id = 0; id < nElements; ++id)
    {
        destData[id] = sourceData[id];
    }
    return services::Status();
}


template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::signFlipEigenvectors(NumericTable& eigenvectors) const
{
    size_t nFeatures = eigenvectors.getNumberOfColumns();
    size_t nVectors = eigenvectors.getNumberOfRows();
    WriteRows<algorithmFPType, cpu> eigenvectorsBlock(eigenvectors, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(eigenvectorsBlock);
    algorithmFPType *eigenvectorsData = eigenvectorsBlock.get();
    for (size_t id = 0; id < nVectors; ++id)
    {
        signFlipArray(nFeatures, eigenvectorsData + id * nFeatures);
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCADenseBase<algorithmFPType, cpu>::fillTable(NumericTable& table, algorithmFPType val) const
{
    size_t nElements = table.getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> tableBlock(table, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(tableBlock);
    algorithmFPType *tableData = tableBlock.get();
    for (size_t id = 0; id < nElements; ++id)
    {
        tableData[id] = val;
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
void PCADenseBase<algorithmFPType, cpu>::signFlipArray(size_t size, algorithmFPType *source) const
{
#define FABS(X) ((X) > (algorithmFPType)0 ? (X): -(X));
    algorithmFPType smax = source[0];
    algorithmFPType max = FABS(smax)
        for (size_t id = 1; id < size; ++id)
        {
            algorithmFPType tmp = FABS(source[id]);
            if (tmp > max)
            {
                max = tmp;
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

#endif // __PCA_DENSE_BASE_H__
