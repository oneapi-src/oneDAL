/* file: em_gmm_batch.h */
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
//  Implementation of the EM for GMM interface.
//--
*/

#ifndef __EM_BATCH_
#define __EM_BATCH_

#include "em_gmm_types.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{

/**
 * Allocates memory for storing results of the EM for GMM algorithm
 * \param[in] input     Pointer to the input structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
    size_t nComponents = algParameter->nComponents;

    Argument::set(weights, data_management::SerializationIfacePtr(new data_management::HomogenNumericTable<algorithmFPType>
                  (nComponents, 1,
                   data_management::NumericTable::doAllocate, 0)));
    Argument::set(means, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(
                          nFeatures, nComponents, data_management::NumericTable::doAllocate, 0)));

    data_management::DataCollectionPtr covarianceCollection =
        data_management::DataCollectionPtr(new data_management::DataCollection());
    for(size_t i = 0; i < nComponents; i++)
    {
        covarianceCollection->push_back(data_management::NumericTablePtr(
                                            new data_management::HomogenNumericTable<algorithmFPType>(
                                                nFeatures, nFeatures, data_management::NumericTable::doAllocate, 0)));
    }

    Argument::set(covariances, services::staticPointerCast<data_management::SerializationIface, data_management::DataCollection>
                  (covarianceCollection));
    Argument::set(goalFunction, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(
                          1, 1, data_management::NumericTable::doAllocate, 0)));
    Argument::set(nIterations, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<int>(
                          1, 1, data_management::NumericTable::doAllocate, 0)));
}

} // namespace em_gmm
} // namespace algorithms
} // namespace daal

#endif
