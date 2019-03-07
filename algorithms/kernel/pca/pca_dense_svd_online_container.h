/* file: pca_dense_svd_online_container.h */
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

#ifndef __PCA_DENSE_SVD_ONLINE_CONTAINER_H__
#define __PCA_DENSE_SVD_ONLINE_CONTAINER_H__

#include "kernel.h"
#include "pca_online.h"
#include "pca_dense_svd_online_kernel.h"
#include "pca_dense_svd_container.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{

template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, svdDense, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDOnlineKernel, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, svdDense, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, svdDense, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    internal::InputDataType dtype = getInputDataType(input);

    PartialResult<svdDense> *partialResult = static_cast<PartialResult<svdDense> *>(_pres);

    NumericTablePtr data = input->get(pca::data);

    NumericTablePtr nObservations = partialResult->get(pca::nObservationsSVD);
    NumericTablePtr sumSquaresSVD = partialResult->get(pca::sumSquaresSVD);
    NumericTablePtr sumSVD = partialResult->get(pca::sumSVD);

    DataCollectionPtr rCollection = partialResult->get(auxiliaryData);
    size_t nFeatures = sumSquaresSVD.get()->getNumberOfColumns();
    services::Status s;
    NumericTablePtr auxiliaryTable = HomogenNumericTable<algorithmFPType>::create(nFeatures, nFeatures, NumericTableIface::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);
    rCollection->push_back(auxiliaryTable);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType),
        compute, dtype, data, *nObservations, *auxiliaryTable, *sumSVD, *sumSquaresSVD);
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, svdDense, cpu>::finalizeCompute()
{
    Input *input = static_cast<Input *>(_in);
    internal::InputDataType dtype = getInputDataType(input);

    DataCollectionPtr rCollection;
    Result *result = static_cast<Result *>(_res);

    PartialResult<svdDense> *partialResult = static_cast<PartialResult<svdDense> *>(_pres);

    NumericTablePtr nObservations = partialResult->get(pca::nObservationsSVD);

    rCollection = partialResult->get(auxiliaryData);

    NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), finalizeMerge,
        dtype, nObservations, *eigenvalues, *eigenvectors, rCollection);
}

}
}
} // namespace daal
#endif
