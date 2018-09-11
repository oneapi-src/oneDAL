/* file: pca_dense_svd_base.h */
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
    nonNormalizedDataset = 0,   /*!< Original, non-normalized data set */
    normalizedDataset    = 1,   /*!< Normalized data set whose feature vectors have zero average and unit variance */
    correlation          = 2    /*!< Correlation matrix */
};


template <typename algorithmFPType, CpuType cpu>
class PCASVDKernelBase : public PCADenseBase<algorithmFPType, cpu>
{
public:
    PCASVDKernelBase() {}
    virtual ~PCASVDKernelBase() {}

protected:
    services::Status scaleSingularValues(data_management::NumericTable& eigenvaluesTable, size_t nVectors);
};

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDKernelBase<algorithmFPType, cpu>::scaleSingularValues(NumericTable& eigenvaluesTable, size_t nVectors)
{
    const size_t nFeatures = eigenvaluesTable.getNumberOfColumns();
    daal::internal::WriteRows<algorithmFPType, cpu> block(eigenvaluesTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(block);
    algorithmFPType *eigenvalues = block.get();

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
