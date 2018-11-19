/* file: pca_dense_svd_distr_step2_container.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_DISTR_STEP2_CONTAINER_H__
#define __PCA_DENSE_SVD_DISTR_STEP2_CONTAINER_H__

#include "kernel.h"
#include "pca_distributed.h"
#include "pca_dense_svd_distr_step2_kernel.h"
#include "pca_dense_svd_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDStep2MasterKernel, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::compute()
{
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::finalizeCompute()
{
    Result *result = static_cast<Result *>(_res);

    DistributedInput<svdDense> *input = static_cast<DistributedInput<svdDense> *>(_in);
    PartialResult<svdDense> *partialResult = static_cast<PartialResult<svdDense> *>(_pres);

    data_management::DataCollectionPtr inputPartialResults = input->get(pca::partialResults);

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    daal::services::Environment::env &env = *_env;

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::PCASVDStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), finalizeMerge,
        internal::nonNormalizedDataset, inputPartialResults, *eigenvalues, *eigenvectors);

    inputPartialResults->clear();
    return s;
}

}
}
} // namespace daal
#endif
