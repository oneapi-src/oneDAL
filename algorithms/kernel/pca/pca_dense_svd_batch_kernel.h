/* file: pca_dense_svd_batch_kernel.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_KERNEL_H__
#define __PCA_DENSE_SVD_BATCH_KERNEL_H__

#include "pca_batch.h"
#include "pca_types.h"
#include "svd/svd_dense_default_kernel.h"

#include "pca_dense_svd_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
class PCASVDBatchKernel : public PCASVDKernelBase<algorithmFPType, cpu>
{
public:
    PCASVDBatchKernel() {};

    services::Status compute(InputDataType type, const data_management::NumericTablePtr &data,
        data_management::NumericTable &eigenvalues, data_management::NumericTable& eigenvectors);

    services::Status compute(InputDataType type,
            data_management::NumericTable& data,
            const BatchParameter<algorithmFPType, svdDense>* parameter,
            data_management::NumericTable& eigenvalues,
            data_management::NumericTable& eigenvectors,
            data_management::NumericTable& means,
            data_management::NumericTable& variances);

protected:
    services::Status normalizeDataset(const data_management::NumericTablePtr& data, data_management::NumericTablePtr& normalizedData);

    services::Status decompose(const NumericTable *normalizedDataTable, data_management::NumericTable& eigenvalues,
        data_management::NumericTable& eigenvectors);
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
