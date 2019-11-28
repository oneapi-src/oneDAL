/* file: em_gmm_batch.h */
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
//  Implementation of the EM for GMM interface.
//--
*/

#ifndef __EM_BATCH_
#define __EM_BATCH_

#include "em_gmm_types.h"

using namespace daal::data_management;

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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    Input * algInput               = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);

    size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
    size_t nComponents = algParameter->nComponents;

    services::Status status;

    set(weights, HomogenNumericTable<algorithmFPType>::create(nComponents, 1, NumericTable::doAllocate, 0, &status));
    set(means, HomogenNumericTable<algorithmFPType>::create(nFeatures, nComponents, NumericTable::doAllocate, 0, &status));

    DataCollectionPtr covarianceCollection = DataCollectionPtr(new DataCollection());
    for (size_t i = 0; i < nComponents; i++)
    {
        if (algParameter->covarianceStorage == diagonal)
        {
            covarianceCollection->push_back(HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, 0, &status));
        }
        else
        {
            covarianceCollection->push_back(HomogenNumericTable<algorithmFPType>::create(nFeatures, nFeatures, NumericTable::doAllocate, 0, &status));
        }
    }
    set(covariances, covarianceCollection);

    set(goalFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, 0, &status));
    set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, 0, &status));
    return status;
}

} // namespace em_gmm
} // namespace algorithms
} // namespace daal

#endif
