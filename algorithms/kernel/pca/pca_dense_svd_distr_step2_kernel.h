/* file: pca_dense_svd_distr_step2_kernel.h */
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

#ifndef __PCA_DENSE_SVD_DISTR_STEP2_KERNEL_H__
#define __PCA_DENSE_SVD_DISTR_STEP2_KERNEL_H__

#include "pca_distributed.h"
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
class PCASVDStep2MasterKernel : public PCASVDKernelBase<algorithmFPType, cpu>
{
public:
    PCASVDStep2MasterKernel() {};
    services::Status finalizeMerge(InputDataType type, const data_management::DataCollectionPtr &inputPartialResults,
        data_management::NumericTable &eigenvalues, data_management::NumericTable &eigenvectors);
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
