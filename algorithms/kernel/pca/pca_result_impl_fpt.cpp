/* file: pca_result_impl_fpt.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "pca_result_impl.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface2
{

/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] input         Pointer to an object containing input data
 * \param[in] nComponents   Number of components
 * \param[in] resultsToCompute     Results to compute
 * \return Status of computations
 */
template<typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::Input *input, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    const InputIface *in = static_cast<const InputIface *>(input);
    size_t nFeatures = in->getNFeatures();

    return allocate<algorithmFPType>(nFeatures, nComponents, resultsToCompute);
}

/**
 * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
 * \param[in] partialResult   Partial result
 * \param[in] nComponents     Number of components
 * \param[in] resultsToCompute     Results to compute
 * \return Status of computations
 */
template<typename algorithmFPType>
services::Status ResultImpl::allocate(const daal::algorithms::PartialResult *partialResult, size_t nComponents, DAAL_UINT64 resultsToCompute)
{
    const PartialResultBase *partialRes = static_cast<const PartialResultBase *>(partialResult);
    size_t nFeatures = partialRes->getNFeatures();

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
    services::Status status;

    if (nComponents == 0)
    {
        nComponents = nFeatures;
    }

    setTable(eigenvalues, data_management::HomogenNumericTable<algorithmFPType>::create(nComponents, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);

    setTable(eigenvectors, data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, nComponents, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);
    if (resultsToCompute & eigenvalue)
    {
        isWhitening = true;
    }
    if (resultsToCompute & mean)
    {
        setTable(means, data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTableIface::doAllocate, 0, &status));
        DAAL_CHECK_STATUS_VAR(status);
    }
    if (resultsToCompute & variance)
    {
        setTable(variances, data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTableIface::doAllocate, 0, &status));
        DAAL_CHECK_STATUS_VAR(status);
    }
    return status;
}

template services::Status ResultImpl::allocate<DAAL_FPTYPE>(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, size_t nComponents, DAAL_UINT64 resultsToCompute);
template services::Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *partialResult, size_t nComponents, DAAL_UINT64 resultsToCompute);

}// interface2
}// namespace pca
}// namespace algorithms
}// namespace daal
