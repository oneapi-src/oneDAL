/* file: implicit_als_predict_ratings_result_fpt.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
/**
 * Allocates memory to store the result of the rating prediction stage of the implicit ALS algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const InputIface * algInput = static_cast<const InputIface *>(input);

    size_t nUsers = algInput->getNumberOfUsers();
    size_t nItems = algInput->getNumberOfItems();
    Status st;
    set(prediction, HomogenNumericTable<algorithmFPType>::create(nItems, nUsers, NumericTableIface::doAllocate, &st));
    return st;
}

template Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                          const int method);

} // namespace interface1
} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
