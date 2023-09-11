/* file: cholesky_batch.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of cholesky algorithm and types methods.
//--
*/
#ifndef __CHOLESKY_BATCH__
#define __CHOLESKY_BATCH__

#include "algorithms/cholesky/cholesky_types.h"

using namespace daal::data_management;
namespace daal
{
namespace algorithms
{
namespace cholesky
{
/**
 * Allocates memory to store the results of Cholesky decomposition
 * \param[in] input  Pointer to the input structure
 * \param[in] par    Pointer to the parameter structure
 * \param[in] method Computation method of the algorithm
 */
template <typename algFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    Input * algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    services::Status status;
    set(choleskyFactor, HomogenNumericTable<algFPType>::create(nFeatures, nFeatures, NumericTable::doAllocate, &status));
    return status;
}

} // namespace cholesky
} // namespace algorithms
} // namespace daal

#endif
