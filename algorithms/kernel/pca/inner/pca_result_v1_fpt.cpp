/* file: pca_result_v1_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "algorithms/pca/pca_types.h"
#include "pca/inner/pca_result_v1.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] input Pointer to an object containing input data
 * \param[in] parameter Algorithm parameter
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, daal::algorithms::Parameter * parameter, const Method method)
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, services::ErrorNullPtr);

    return impl->allocate<algorithmFPType>(input);
}

/**
 * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
 * \param[in] parameter Parameter of the algorithm
 * \param[in] method        Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter,
                                              const Method method)
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, services::ErrorNullPtr);

    return impl->allocate<algorithmFPType>(partialResult);
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, daal::algorithms::Parameter * parameter,
                                                                    const Method method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * partialResult,
                                                                    daal::algorithms::Parameter * parameter, const Method method);

/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] input Pointer to an object containing input data
 * \param[in] parameter Algorithm parameter
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::Input * input)
{
    const InputIface * in = static_cast<const InputIface *>(input);
    DAAL_CHECK(in, services::ErrorNullPtr);
    size_t nFeatures = in->getNFeatures();

    return allocate<algorithmFPType>(nFeatures);
}

/**
 * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
 * \param[in] parameter Parameter of the algorithm
 * \param[in] method        Computation method
 */
template <typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::PartialResult * partialResult)
{
    const PartialResultBase * partialRes = static_cast<const PartialResultBase *>(partialResult);
    DAAL_CHECK(partialRes, services::ErrorNullPtr);
    size_t nFeatures = partialRes->getNFeatures();

    return allocate<algorithmFPType>(nFeatures);
}

/**
* Allocates memory for storing partial results of the PCA algorithm
* \param[in] nFeatures Number of features
* \return Status of computations
*/
template <typename algorithmFPType>
services::Status ResultImpl::allocate(size_t nFeatures)
{
    services::Status status;

    setTable(eigenvalues,
             data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);

    setTable(eigenvectors, data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, nFeatures,
                                                                                         data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template services::Status ResultImpl::allocate<DAAL_FPTYPE>(size_t nFeatures);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * partialResult);

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
