/* file: pca_transform_batch_fpt.cpp */
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
//  Implementation of the regression algorithm interface
//--
*/

#include "algorithms/pca/transform/pca_transform_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "src/services/daal_strings.h"

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
Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const Input * in            = static_cast<const Input *>(input);
    const Parameter * parameter = static_cast<const Parameter *>(par);

    data_management::NumericTablePtr dataPtr         = in->get(data);
    data_management::NumericTablePtr eigenvectorsPtr = in->get(eigenvectors);
    DAAL_CHECK_EX(dataPtr.get(), ErrorNullInputNumericTable, ArgumentName, dataStr())
    DAAL_CHECK_EX(eigenvectorsPtr.get(), ErrorNullInputNumericTable, ArgumentName, eigenvectorsStr())

    size_t nInputs     = dataPtr->getNumberOfRows();
    size_t nComponents = parameter->nComponents == 0 ? eigenvectorsPtr->getNumberOfRows() : parameter->nComponents;

    services::Status status;

    NumericTablePtr transformedDataNT;

    transformedDataNT = HomogenNumericTable<algorithmFPType>::create(nComponents, nInputs, NumericTable::doAllocate, &status);

    DAAL_CHECK_STATUS_VAR(status);

    set(transformedData, transformedDataNT);

    return status;
}

template Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
