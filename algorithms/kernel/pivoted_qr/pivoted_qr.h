/* file: pivoted_qr.h */
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
//  Definition of Pivoted QR common types.
//--
*/
#ifndef __PIVOTED_QR_BATCH__
#define __PIVOTED_QR_BATCH__

#include "pivoted_qr_types.h"

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{

/**
 * Allocates memory for storing final results of the pivoted QR algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    size_t m = static_cast<const Input *>(input)->get(data)->getNumberOfColumns();
    size_t n = static_cast<const Input *>(input)->get(data)->getNumberOfRows();

    Argument::set(matrixQ, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
    Argument::set(matrixR, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
    Argument::set(permutationMatrix, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<size_t>(m, 1, data_management::NumericTable::doAllocate, 0)));
}

}// namespace pivoted_qr
}// namespace algorithms
}// namespace daal

#endif
