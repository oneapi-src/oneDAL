/* file: pca_partialresult_correlation.cpp */
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

PartialResult<correlationDense>::PartialResult() : PartialResultBase(3) {};

/**
 * Gets partial results of the PCA Correlation algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr PartialResult<correlationDense>::get(PartialCorrelationResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

size_t PartialResult<correlationDense>::getNFeatures() const { return get(sumCorrelation)->getNumberOfColumns(); }

/**
 * Sets partial result of the PCA Correlation algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void PartialResult<correlationDense>::set(const PartialCorrelationResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
* Checks partial results of the PCA Correlation algorithm
* \param[in] input      %Input object of the algorithm
* \param[in] parameter  Algorithm %parameter
* \param[in] method     Computation method
*/
void PartialResult<correlationDense>::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const InputIface *in = static_cast<const InputIface *>(input);
    DAAL_CHECK(!in->isCorrelation(), ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
    checkImpl(in->getNFeatures());
}

/**
* Checks partial results of the PCA Ccorrelation algorithm
* \param[in] par        Algorithm %parameter
* \param[in] method     Computation method
*/
void PartialResult<correlationDense>::check(const daal::algorithms::Parameter *par, int method) const
{
    checkImpl(0);
}

void PartialResult<correlationDense>::checkImpl(size_t nFeatures) const
{
    int csrLayout = (int)NumericTableIface::csrArray;
    int packedLayouts = packed_mask;
    NumericTablePtr sumCorrelation = get(pca::sumCorrelation);
    if(!checkNumericTable(get(pca::nObservationsCorrelation).get(), this->_errors.get(), nObservationsCorrelationStr(), csrLayout, 0, 1, 1)) { return; }
    if(!checkNumericTable(sumCorrelation.get(), this->_errors.get(), sumCorrelationStr(), packedLayouts, 0, nFeatures, 1)) { return; }
    nFeatures = sumCorrelation->getNumberOfColumns();
    if(!checkNumericTable(get(pca::crossProductCorrelation).get(), this->_errors.get(), crossProductCorrelationStr(), packedLayouts, 0, nFeatures, nFeatures)) { return; }
}

/**
 * Initialize partial results
 * \param[in]       input     Input objects for the PCA algorithm
 * \param[in,out]   pres      Partial results of the PCA algorithm
 * \return                    Initialized partial results
 */
void DefaultPartialResultsInit<correlationDense>::operator()(const Input &input, services::SharedPtr<PartialResult<correlationDense> > &pres)
{
    setToZero(pres->get(pca::nObservationsCorrelation).get());
    setToZero(pres->get(pca::sumCorrelation).get());
    setToZero(pres->get(pca::crossProductCorrelation).get());
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
