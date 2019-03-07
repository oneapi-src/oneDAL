/* file: zscore_result.cpp */
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
//  Implementation of zscore internal result.
//--
*/

#include "zscore_result.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface2
{

/**
* Checks the correctness of the Result object
* \param[in] in     Pointer to the input object
* \param[in] par    Pointer to the parameter object
*/
Status ResultImpl::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par) const
{
    Status status = interface1::ResultImpl::check(in);
    DAAL_CHECK_STATUS_VAR(status);

    const interface1::Input *input = static_cast<const interface1::Input *>(in);
    DAAL_CHECK(input, ErrorNullInput);

    NumericTablePtr dataTable = input->get(zscore::data);
    DAAL_CHECK(dataTable, ErrorNullInputNumericTable);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    const size_t nVectors = dataTable->getNumberOfRows();

    const interface2::BaseParameter* parameter = static_cast<const BaseParameter*>(par);
    if (parameter->resultsToCompute & mean)
    {
        DAAL_CHECK_STATUS(status, checkNumericTable(NumericTable::cast(get(means)).get(), meansStr(), packed_mask, 0, nFeatures, 1))
    }
    if (parameter->resultsToCompute & variance)
    {
        DAAL_CHECK_STATUS(status, checkNumericTable(NumericTable::cast(get(variances)).get(), variancesStr(), packed_mask, 0, nFeatures, 1))
    }
    return status;
}

}// namespace interface2

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
