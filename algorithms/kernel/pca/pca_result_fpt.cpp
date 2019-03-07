/* file: pca_result_fpt.cpp */
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
#include "algorithms/pca/pca_types.h"
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
 * \param[in] input Pointer to an object containing input data
 * \param[in] parameter Algorithm parameter
 * \param[in] method Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, daal::algorithms::Parameter *parameter, const Method method)
{
    size_t nComponents = 0;
    DAAL_UINT64 resultsToCompute = eigenvalue;

    const auto* par = dynamic_cast<const BaseBatchParameter*>(parameter);
    if (par != NULL)
    {
        nComponents = par->nComponents;
        resultsToCompute = par->resultsToCompute;
    }

    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, services::ErrorNullPtr);

    return impl->allocate<algorithmFPType>(input, nComponents, resultsToCompute);
}

/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] partialResult Pointer to an object containing input data
 * \param[in] parameter Parameter of the algorithm
 * \param[in] method        Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const Method method)
{
    size_t nComponents = 0;
    DAAL_UINT64 resultsToCompute = eigenvalue;

    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, services::ErrorNullPtr);

    return impl->allocate<algorithmFPType>(partialResult, nComponents, resultsToCompute);
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, daal::algorithms::Parameter *parameter, const Method method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const Method method);

}// interface2
}// namespace pca
}// namespace algorithms
}// namespace daal
