/* file: em_gmm_init.h */
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
//  Implementation of the EM for GMM initialization interface.
//--
*/

#ifndef __EM_INIT_
#define __EM_INIT_

#include "em_gmm_init_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{

/**
 * Allocates memory for storing initial values for results of the EM for GMM algorithm
 * \param[in] input        Pointer to the input structure
 * \param[in] parameter    Pointer to the parameter structure
 * \param[in] method       Method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));

    size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
    size_t nComponents = algParameter->nComponents;

    services::Status status;
    Argument::set(weights, HomogenNumericTable<algorithmFPType>::create(nComponents, 1, NumericTable::doAllocate, 0, &status));
    Argument::set(means, HomogenNumericTable<algorithmFPType>::create(nFeatures, nComponents, NumericTable::doAllocate, 0, &status));

    DataCollectionPtr covarianceCollection = DataCollectionPtr(new DataCollection());
    for(size_t i = 0; i < nComponents; i++)
    {
        if(algParameter->covarianceStorage == em_gmm::diagonal)
        {
            covarianceCollection->push_back(HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, 0, &status));
        }
        else
        {
            covarianceCollection->push_back(HomogenNumericTable<algorithmFPType>::create(nFeatures, nFeatures, NumericTable::doAllocate, 0, &status));
        }
    }
    Argument::set(covariances, services::staticPointerCast<SerializationIface, DataCollection>(covarianceCollection));
    return status;
}

} // namespace init
} // namespace em_gmm
} // namespace algorithms
} // namespace daal

#endif
