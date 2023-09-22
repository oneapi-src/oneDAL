/* file: pca_result_impl_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "src/algorithms/pca/pca_result_impl.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] input         Pointer to an object containing input data
 * \param[in] nComponents   Number of components
 * \param[in] resultsToCompute     Results to compute
 * \return Status of computations
 */
template <typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::Input * input, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    const InputIface * in = static_cast<const InputIface *>(input);
    size_t nFeatures      = in->getNFeatures();

    return allocate<algorithmFPType>(nFeatures, nComponents, resultsToCompute);
}

/**
 * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
 * \param[in] partialResult   Partial result
 * \param[in] nComponents     Number of components
 * \param[in] resultsToCompute     Results to compute
 * \return Status of computations
 */
template <typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::PartialResult * partialResult, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    const PartialResultBase * partialRes = static_cast<const PartialResultBase *>(partialResult);
    size_t nFeatures                     = partialRes->getNFeatures();

    return allocate<algorithmFPType>(nFeatures, nComponents, resultsToCompute);
}

/**
* Allocates memory for storing partial results of the PCA algorithm
* \param[in] nFeatures      Number of features
* \param[in] nComponents    Number of components
* \param[in] resultsToCompute     Results to compute
* \return Status of computations
*/
template <typename algorithmFPType>
services::Status ResultImpl::allocate(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        return __allocate__impl<algorithmFPType, data_management::HomogenNumericTable<algorithmFPType> >(nFeatures, nComponents, resultsToCompute);
    }
    else
    {
        return __allocate__impl<algorithmFPType, data_management::internal::SyclHomogenNumericTable<algorithmFPType> >(nFeatures, nComponents,
                                                                                                                       resultsToCompute);
    }
}

template <typename algorithmFPType, typename NumericTableType>
services::Status ResultImpl::__allocate__impl(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    services::Status status;

    if (nComponents == 0)
    {
        nComponents = nFeatures;
    }

    setTable(eigenvalues, NumericTableType::create(nComponents, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);

    setTable(eigenvectors, NumericTableType::create(nFeatures, nComponents, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);
    if (resultsToCompute & eigenvalue)
    {
        isWhitening = true;
    }
    if (resultsToCompute & mean)
    {
        setTable(means, NumericTableType::create(nFeatures, 1, data_management::NumericTableIface::doAllocate, 0, &status));
        DAAL_CHECK_STATUS_VAR(status);
    }
    if (resultsToCompute & variance)
    {
        setTable(variances, NumericTableType::create(nFeatures, 1, data_management::NumericTableIface::doAllocate, 0, &status));
        DAAL_CHECK_STATUS_VAR(status);
    }
    return status;
}

template services::Status ResultImpl::allocate<DAAL_FPTYPE>(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, size_t nComponents, DAAL_UINT64 resultsToCompute);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * partialResult, size_t nComponents,
                                                            DAAL_UINT64 resultsToCompute);

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal
