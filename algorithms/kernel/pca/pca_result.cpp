/* file: pca_result.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

Result::Result() : daal::algorithms::Result(2) {};

/**
* Gets the results of the PCA algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}
/**
 * Sets results of the PCA algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}
/**
* Checks the results of the PCA algorithm
* \param[in] _input  %Input object of algorithm
* \param[in] par     Algorithm %parameter
* \param[in] method  Computation  method
*/
void Result::check(const daal::algorithms::Input *_input, const daal::algorithms::Parameter *par, int method) const
{
    const InputIface *input = static_cast<const InputIface *>(_input);
    checkImpl(input->getNFeatures());
}
/**
* Checks the results of the PCA algorithm
* \param[in] pr             Partial results of the algorithm
* \param[in] method         Computation method
* \param[in] parameter      Algorithm %parameter
*/
void Result::check(const daal::algorithms::PartialResult *pr, const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(0);
}
/**
* Checks the results of the PCA algorithm
* \param[in] pr             Partial results of the algorithm
* \param[in] method         Computation method
* \param[in] parameter      Algorithm %parameter
*/
void Result::checkImpl(size_t nFeatures) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfOutputNumericTables);
    int packedLayouts = packed_mask;
    if(!checkNumericTable(get(eigenvalues).get(), this->_errors.get(), eigenvaluesStr(), packedLayouts, 0, nFeatures, 1)) { return; }
    nFeatures = get(eigenvalues)->getNumberOfColumns();
    if(!checkNumericTable(get(eigenvectors).get(), this->_errors.get(), eigenvectorsStr(), packedLayouts, 0, nFeatures, nFeatures)) { return; }
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
