/* file: zscore_fpt.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "zscore_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface1
{
/**
 * Allocates memory to store final results of the z-score normalization algorithms
 * \param[in] input     Input objects for the z-score normalization algorithm
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const int method)
{
    const Input *in = static_cast<const Input *>(input);

    size_t nFeatures = in->get(data)->getNumberOfColumns();
    size_t nVectors = in->get(data)->getNumberOfRows();

    Argument::set(normalizedData, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nVectors,
                                                                                data_management::NumericTable::doAllocate)));
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const int method);

}// namespace interface1
}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
