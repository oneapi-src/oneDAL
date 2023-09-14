/* file: pca_result.cpp */
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

#include "algorithms/pca/pca_types.h"
#include "src/algorithms/pca/pca_result_impl.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_PCA_RESULT_ID);

Result::Result(const Result & o)
{
    ResultImpl * pImpl = dynamic_cast<ResultImpl *>(getStorage(o).get());
    DAAL_ASSERT(pImpl);
    if (pImpl != NULL)
    {
        Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(*pImpl)));
    }
    else
    {
        Result();
    }
}

Result::Result() : daal::algorithms::Result(lastResultId + 1)
{
    Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(lastResultId + 1)));
};

/**
* Gets the results of the PCA algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
NumericTablePtr Result::get(ResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
* Gets the results collection of the PCA algorithm
* \param[in] id    Identifier of the results collection
* \return          PCA results collection
*/
data_management::KeyValueDataCollectionPtr Result::get(ResultCollectionId id) const
{
    KeyValueDataCollectionPtr pResultsCollection = KeyValueDataCollectionPtr(new KeyValueDataCollection());
    KeyValueDataCollection & resultsCollection   = *pResultsCollection;
    resultsCollection[mean]                      = get(means);
    resultsCollection[variance]                  = get(variances);
    resultsCollection[eigenvalue]                = NumericTablePtr();
    auto impl                                    = ResultImpl::cast(getStorage(*this));
    if (impl)
    {
        if (impl->isWhitening) resultsCollection[eigenvalue] = get(eigenvalues);
    }

    return pResultsCollection;
}

/**
* Sets the results collection of the PCA algorithm
* only not NULL tables from collection collection will be set to result
* \param[in] id          Identifier of the results collection
* \param[in] collection  PCA results collection
*/
void Result::set(ResultCollectionId id, data_management::KeyValueDataCollectionPtr & collection)
{
    if (collection.get() != NULL)
    {
        KeyValueDataCollection & resultsCollection = *collection;
        if (resultsCollection[mean].get() != NULL) set(means, NumericTable::cast(resultsCollection[mean]));
        if (resultsCollection[variance].get() != NULL) set(variances, NumericTable::cast(resultsCollection[variance]));
        if (resultsCollection[eigenvalue].get() != NULL) set(eigenvalues, NumericTable::cast(resultsCollection[eigenvalue]));
    }
}

/**
 * Sets results of the PCA algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
* Checks the partial results of the PCA algorithm
* \param[in] pr             Partial results of the algorithm
* \param[in] method         Computation method
* \param[in] parameter      Algorithm %parameter
*
* \return Status
*/
services::Status Result::check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * parameter, int method) const
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, ErrorNullPtr);

    return impl->check(0, lastResultId + 1);
}

/**
* Checks the results of the PCA algorithm implementation
* \param[in] nFeatures             Number of features
* \param[in] resultsToCompute      Results to compute
*
* \return Status
*/
services::Status Result::checkImpl(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute) const
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, ErrorNullPtr);

    Status s = impl->check(nFeatures, nComponents, lastResultId + 1);
    if (resultsToCompute & mean)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(means).get(), meansStr(), packed_mask, 0, nFeatures, 1));
    }
    if (resultsToCompute & variance)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(variances).get(), variancesStr(), packed_mask, 0, nFeatures, 1));
    }

    return s;
}

/**
* Checks the results of the PCA algorithm
* \param[in] input       %Input object of algorithm
* \param[in] parameter   Algorithm %parameter
* \param[in] method      Computation  method
*
* \return Status
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    size_t nComponents           = 0;
    DAAL_UINT64 resultsToCompute = eigenvalue;

    const auto * par = dynamic_cast<const BaseBatchParameter *>(parameter);
    if (par != NULL)
    {
        nComponents      = par->nComponents;
        resultsToCompute = par->resultsToCompute;
    }

    const interface1::InputIface * in = static_cast<const interface1::InputIface *>(input);
    DAAL_CHECK(in, ErrorNullPtr);

    return checkImpl(in->getNFeatures(), nComponents, resultsToCompute);
}

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal
