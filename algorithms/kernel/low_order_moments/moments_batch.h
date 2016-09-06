/* file: moments_batch.h */
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
//  Implementation of LowOrderMoments algorithm and types methods.
//--
*/
#ifndef __MOMENTS_BATCH__
#define __MOMENTS_BATCH__

#include "low_order_moments_types.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{

/**
 * Allocates memory for storing final results of the low order %moments algorithm
 * \param[in] input     Pointer to the structure with result objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfColumns();

    for(size_t i = 0; i < nResults; i++)
    {
        Argument::set(i, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, 1, data_management::NumericTable::doAllocate)));
    }
}

/**
 * Allocates memory for storing final results of the low order %moments algorithm
 * \param[in] partialResult     Pointer to the structure with partial result objects
 * \param[in] parameter         Pointer to the structure of algorithm parameters
 * \param[in] method            Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method)
{
    size_t nFeatures = static_cast<const PartialResult *>(partialResult)->getNumberOfColumns();

    for(size_t i = 0; i < nResults; i++)
    {
        Argument::set(i, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, 1, data_management::NumericTable::doAllocate)));
    }
}

}// namespace low_order_moments
}// namespace algorithms
}// namespace daal

#endif
