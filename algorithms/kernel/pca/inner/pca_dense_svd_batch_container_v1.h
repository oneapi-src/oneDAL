/* file: pca_dense_svd_batch_container_v1.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_CONTAINER_V1_H__
#define __PCA_DENSE_SVD_BATCH_CONTAINER_V1_H__

#include "pca/inner/pca_batch_v1.h"
#include "kernel.h"
#include "pca_dense_svd_batch_kernel.h"
#include "pca_dense_svd_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDBatchKernel, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
Status BatchContainer<algorithmFPType, svdDense, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    internal::InputDataType dtype = getInputDataType(input);

    data_management::NumericTablePtr data = input->get(pca::data);
    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, dtype, data, *eigenvalues, *eigenvectors);
}

} // interface1
}
}
} // namespace daal
#endif
