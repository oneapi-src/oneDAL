/* file: pca_dense_svd_online_container.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_ONLINE_CONTAINER_H__
#define __PCA_DENSE_SVD_ONLINE_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_online.h"
#include "src/algorithms/pca/pca_dense_svd_online_kernel.h"
#include "src/algorithms/pca/pca_dense_svd_container.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{
template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, svdDense, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
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
    Input * input                 = static_cast<Input *>(_in);
    internal::InputDataType dtype = getInputDataType(input);

    PartialResult<svdDense> * partialResult = static_cast<PartialResult<svdDense> *>(_pres);

    NumericTablePtr data = input->get(pca::data);

    NumericTablePtr nObservations = partialResult->get(pca::nObservationsSVD);
    NumericTablePtr sumSquaresSVD = partialResult->get(pca::sumSquaresSVD);
    NumericTablePtr sumSVD        = partialResult->get(pca::sumSVD);

    DataCollectionPtr rCollection = partialResult->get(auxiliaryData);
    size_t nFeatures              = sumSquaresSVD.get()->getNumberOfColumns();
    services::Status s;
    NumericTablePtr auxiliaryTable = HomogenNumericTable<algorithmFPType>::create(nFeatures, nFeatures, NumericTableIface::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);
    rCollection->push_back(auxiliaryTable);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, dtype, data, *nObservations,
                       *auxiliaryTable, *sumSVD, *sumSquaresSVD);
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, svdDense, cpu>::finalizeCompute()
{
    Input * input                 = static_cast<Input *>(_in);
    internal::InputDataType dtype = getInputDataType(input);

    DataCollectionPtr rCollection;
    Result * result = static_cast<Result *>(_res);

    PartialResult<svdDense> * partialResult = static_cast<PartialResult<svdDense> *>(_pres);

    NumericTablePtr nObservations = partialResult->get(pca::nObservationsSVD);

    rCollection = partialResult->get(auxiliaryData);

    NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), finalizeMerge, dtype, nObservations, *eigenvalues,
                       *eigenvectors, rCollection);
}

} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
