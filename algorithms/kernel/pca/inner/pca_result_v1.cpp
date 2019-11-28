/* file: pca_result_v1.cpp */
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

#include "pca/inner/pca_result_v1.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_PCA_RESULT_ID);

Result::Result(const Result & o)
{
    ResultImpl * pImpl = dynamic_cast<ResultImpl *>(getStorage(o).get());
    DAAL_ASSERT(pImpl);
    Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(*pImpl)));
}

Result::Result() : daal::algorithms::Result(lastResultId + 1)
{
    Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(lastResultId + 1)));
}

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
 * Sets results of the PCA algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
* Checks the results of the PCA algorithm
* \param[in] _input  %Input object of algorithm
* \param[in] par     Algorithm %parameter
* \param[in] method  Computation  method
*
* \return Status
*/
services::Status Result::check(const daal::algorithms::Input * _input, const daal::algorithms::Parameter * par, int method) const
{
    const interface1::InputIface * input = static_cast<const interface1::InputIface *>(_input);
    DAAL_CHECK(input, ErrorNullPtr);

    return checkImpl(input->getNFeatures());
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
    return checkImpl(0);
}

/**
* Checks the results of the PCA algorithm implementation
* \param[in] nFeatures      Number of features
* \param[in] nTables        Number of tables
*
* \return Status
*/
services::Status Result::checkImpl(size_t nFeatures) const
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, ErrorNullPtr);

    return impl->check(nFeatures, lastResultId + 1);
}

/**
* Checks the results of the PCA algorithm implementation
* \param[in] nFeatures      Number of features
* \param[in] nTables        Number of tables
*
* \return Status
*/
services::Status ResultImpl::check(size_t nFeatures, size_t nTables) const
{
    DAAL_CHECK(size() == nTables, ErrorIncorrectNumberOfOutputNumericTables);
    const int packedLayouts = packed_mask;
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(NumericTable::cast(get(eigenvalues)).get(), eigenvaluesStr(), packedLayouts, 0, nFeatures, 1));

    auto pEigenvalues = NumericTable::cast(get(eigenvalues));
    DAAL_CHECK(pEigenvalues, ErrorNullNumericTable);

    auto pEigenvectors = NumericTable::cast(get(eigenvectors));
    DAAL_CHECK(pEigenvectors, ErrorNullNumericTable);

    nFeatures = pEigenvalues->getNumberOfColumns();
    return checkNumericTable(pEigenvectors.get(), eigenvectorsStr(), packedLayouts, 0, nFeatures, nFeatures);
}

void ResultImpl::setTable(size_t key, data_management::NumericTablePtr table)
{
    (*this)[key] = table;
}

ResultImpl::ResultImpl(const size_t n) : DataCollection(n) {}
ResultImpl::ResultImpl(const ResultImpl & o) : DataCollection(o) {}
ResultImpl::~ResultImpl() {};

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
