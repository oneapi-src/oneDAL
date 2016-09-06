/* file: apriori_batch.h */
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
//  Implementation of the interface for the association rules algorithm
//--
*/
#ifndef __APRIORI_BATCH__
#define __APRIORI_BATCH__

#include "apriori_types.h"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace interface1
{

/**
 * Allocates memory for storing Association Rules algorithm results
 * \param[in] input         Pointer to input structure
 * \param[in] parameter     Pointer to parameter structure
 * \param[in] method        Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));

    Argument::set(largeItemsets,
                  data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
    Argument::set(largeItemsetsSupport,
                  data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));

    if(algParameter->discoverRules)
    {
        Argument::set(antecedentItemsets,
                      data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
        Argument::set(consequentItemsets,
                      data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
        Argument::set(confidence,
                      data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(1, 0, data_management::NumericTableIface::notAllocate)));
    }
}

}// namespace interface1
}// namespace association_rules
}// namespace algorithms
}// namespace daal

#endif
