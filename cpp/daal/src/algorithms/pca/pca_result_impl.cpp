/* file: pca_result_impl.cpp */
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
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
* Checks the results of the PCA algorithm implementation
* \param[in] nFeatures      Number of features
* \param[in] nTables        Number of tables
*
* \return Status
*/
services::Status ResultImpl::check(size_t nFeatures, size_t nComponents, size_t nTables) const
{
    if (nComponents == 0)
    {
        nComponents = nFeatures;
    }

    DAAL_CHECK(size() == nTables, ErrorIncorrectNumberOfOutputNumericTables);
    const int packedLayouts = packed_mask;
    Status status;

    auto pEigenvalues = NumericTable::cast(get(eigenvalues));
    DAAL_CHECK(pEigenvalues, ErrorNullNumericTable);

    auto pEigenvectors = NumericTable::cast(get(eigenvectors));
    DAAL_CHECK(pEigenvectors, ErrorNullNumericTable);

    DAAL_CHECK_STATUS(status, checkNumericTable(pEigenvalues.get(), eigenvaluesStr(), packedLayouts, 0, nComponents, 1));

    DAAL_CHECK_STATUS(status, checkNumericTable(pEigenvectors.get(), eigenvectorsStr(), packedLayouts, 0, nFeatures, nComponents));

    DAAL_CHECK_STATUS_VAR(status);
    // TODO:
    return status;
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

} // namespace pca
} // namespace algorithms
} // namespace daal
