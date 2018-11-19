/* file: pca_result_impl.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{

namespace interface2
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

} // namespace interface2
} // namespace pca
} // namespace algorithms
} // namespace daal
