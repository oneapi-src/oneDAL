/* file: pca_dense_correlation_base.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Declaration of template structs that calculate PCA Correlation.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BASE_H__
#define __PCA_DENSE_CORRELATION_BASE_H__

#include "pca_types.h"
#include "service_lapack.h"
#include "service_defines.h"
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
class PCACorrelationBase : public Kernel
{
public:
    explicit PCACorrelationBase() {};

    virtual ~PCACorrelationBase() {};

protected:
    void computeCorrelationEigenvalues(const data_management::NumericTablePtr correlation,
                                       const data_management::NumericTablePtr eigenvectors,
                                       const data_management::NumericTablePtr eigenvalues);
    void computeEigenvectorsInplace(size_t nFeatures, algorithmFPType *eigenvectors, algorithmFPType *eigenvalues);
    void sortEigenvectorsDescending(size_t nFeatures, algorithmFPType *eigenvectors, algorithmFPType *eigenvalues);

private:
    void copyArray(size_t size, algorithmFPType *source, algorithmFPType *destination);
};

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationBase<algorithmFPType, cpu>::copyArray(size_t size, algorithmFPType *source, algorithmFPType *destination)
{
    if (source != destination)
    {
        for (size_t i = 0; i < size; i++)
        {
            destination[i] = source[i];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationBase<algorithmFPType, cpu>::computeCorrelationEigenvalues(
    const data_management::NumericTablePtr correlation,
    data_management::NumericTablePtr eigenvectors,
    data_management::NumericTablePtr eigenvalues)
{
    using data_management::BlockDescriptor;

    size_t nFeatures = correlation->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> correlationBlock;
    correlation->getBlockOfRows(0, nFeatures, data_management::readOnly, correlationBlock);
    algorithmFPType *correlationArray = correlationBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> eigenvectorsBlock;
    eigenvectors->getBlockOfRows(0, nFeatures, data_management::writeOnly, eigenvectorsBlock);
    algorithmFPType *eigenvectorsArray = eigenvectorsBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> eigenvaluesBlock;
    eigenvalues->getBlockOfRows(0, 1, data_management::writeOnly, eigenvaluesBlock);
    algorithmFPType *eigenvaluesArray = eigenvaluesBlock.getBlockPtr();

    copyArray(nFeatures * nFeatures, correlationArray, eigenvectorsArray);

    computeEigenvectorsInplace(nFeatures, eigenvectorsArray, eigenvaluesArray);
    sortEigenvectorsDescending(nFeatures, eigenvectorsArray, eigenvaluesArray);

    correlation->releaseBlockOfRows(correlationBlock);
    eigenvectors->releaseBlockOfRows(eigenvectorsBlock);
    eigenvalues->releaseBlockOfRows(eigenvaluesBlock);
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationBase<algorithmFPType, cpu>::computeEigenvectorsInplace(size_t nFeatures, algorithmFPType *eigenvectors,
                                                                          algorithmFPType *eigenvalues)
{
    char jobz  = 'V';
    char uplo  = 'U';

    MKL_INT lwork = 2 * nFeatures * nFeatures + 6 * nFeatures + 1;
    MKL_INT liwork = 5 * nFeatures + 3;
    MKL_INT info;

    algorithmFPType *work = (algorithmFPType *)services::daal_malloc(lwork * sizeof(algorithmFPType));
    MKL_INT *iwork = (MKL_INT *)services::daal_malloc(liwork * sizeof(MKL_INT));
    if(work == 0 || iwork == 0)
    {
        services::daal_free(work);
        this->_errors->add(services::ErrorMemoryAllocationFailed);
    }

    Lapack<algorithmFPType, cpu>::xsyevd(&jobz, &uplo, (MKL_INT *)(&nFeatures), eigenvectors, (MKL_INT *)(&nFeatures), eigenvalues,
                        work, &lwork, iwork, &liwork, &info);
    if (info != 0)
    {
        this->_errors->add(services::ErrorPCAFailedToComputeCorrelationEigenvalues);
    }

    services::daal_free(iwork);
    services::daal_free(work);
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationBase<algorithmFPType, cpu>::sortEigenvectorsDescending(size_t nFeatures,
                                                                          algorithmFPType *eigenvectors,
                                                                          algorithmFPType *eigenvalues)
{
    algorithmFPType tmp;
    for(size_t i = 0; i < nFeatures / 2; i++)
    {
        tmp = eigenvalues[i];
        eigenvalues[i] = eigenvalues[nFeatures - 1 - i];
        eigenvalues[nFeatures - 1 - i] = tmp;
    }

    algorithmFPType *eigenvectorTmp = new algorithmFPType[nFeatures];

    for(size_t i = 0; i < nFeatures / 2; i++)
    {
        copyArray(nFeatures, eigenvectors + i * nFeatures, eigenvectorTmp);
        copyArray(nFeatures, eigenvectors + nFeatures * (nFeatures - 1 - i), eigenvectors + i * nFeatures);
        copyArray(nFeatures, eigenvectorTmp, eigenvectors + nFeatures * (nFeatures - 1 - i));
    }

    delete[] eigenvectorTmp;
}

template <ComputeMode mode, typename algorithmFPType, CpuType cpu>
class PCACorrelationKernel : public PCACorrelationBase<algorithmFPType, cpu> {};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
