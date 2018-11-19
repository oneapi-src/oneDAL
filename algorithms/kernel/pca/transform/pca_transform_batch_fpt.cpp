/* file: pca_transform_batch_fpt.cpp */
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
//  Implementation of the regression algorithm interface
//--
*/

#include "algorithms/pca/transform/pca_transform_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "daal_strings.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
using namespace daal::services;
using namespace daal::data_management;

template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    const Parameter* parameter = static_cast<const Parameter *>(par);

    data_management::NumericTablePtr dataPtr = in->get(data);
    data_management::NumericTablePtr eigenvectorsPtr = in->get(eigenvectors);
    DAAL_CHECK_EX(dataPtr.get(), ErrorNullInputNumericTable, ArgumentName, dataStr())
    DAAL_CHECK_EX(eigenvectorsPtr.get(), ErrorNullInputNumericTable, ArgumentName, eigenvectorsStr())

    size_t nInputs             = dataPtr->getNumberOfRows();
    size_t nComponents         = parameter->nComponents == 0 ? eigenvectorsPtr->getNumberOfRows() : parameter->nComponents;

    services::Status status;
    set(transformedData, HomogenNumericTable<algorithmFPType>::create(nComponents, nInputs, NumericTable::doAllocate, &status));

    return status;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
