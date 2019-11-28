/* file: pivoted_qr.h */
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
//  Definition of Pivoted QR common types.
//--
*/
#ifndef __PIVOTED_QR_BATCH__
#define __PIVOTED_QR_BATCH__

#include "pivoted_qr_types.h"

using namespace daal::data_management;

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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    size_t m = static_cast<const Input *>(input)->get(data)->getNumberOfColumns();
    size_t n = static_cast<const Input *>(input)->get(data)->getNumberOfRows();

    set(matrixQ, HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &s));
    set(matrixR, HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &s));
    set(permutationMatrix, HomogenNumericTable<size_t>::create(m, 1, NumericTable::doAllocate, 0, &s));
    return s;
}

} // namespace pivoted_qr
} // namespace algorithms
} // namespace daal

#endif
