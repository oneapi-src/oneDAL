/* file: em_gmm_batch.h */
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
    size_t nComponents = algParameter->nComponents;

    services::Status status;

    set(weights, HomogenNumericTable<algorithmFPType>::create(nComponents, 1, NumericTable::doAllocate, 0, &status));
    set(means, HomogenNumericTable<algorithmFPType>::create(nFeatures, nComponents, NumericTable::doAllocate, 0, &status));

    DataCollectionPtr covarianceCollection = DataCollectionPtr(new DataCollection());
    for(size_t i = 0; i < nComponents; i++)
    {
        if(algParameter->covarianceStorage == diagonal)
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
