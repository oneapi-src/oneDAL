/* file: zscore_result_v1.cpp */
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

#include "zscore_result_v1.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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

namespace interface1
{

/**
* Checks the correctness of the Result object
* \param[in] in     Pointer to the input object
*
* \return Status of computations
*/
services::Status ResultImpl:: check(const daal::algorithms::Input *in) const
{
    const interface1::Input *input = static_cast<const interface1::Input *>(in);
    DAAL_CHECK(input, ErrorNullInput);

    NumericTablePtr dataTable = input->get(zscore::data);
    DAAL_CHECK(dataTable, ErrorNullInputNumericTable);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    const size_t nVectors = dataTable->getNumberOfRows();

    const int unexpectedLayouts = packed_mask;

    return checkNumericTable(NumericTable::cast(get(normalizedData)).get(), normalizedDataStr(), unexpectedLayouts, 0, nFeatures, nVectors);
}

} // interface 1

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
