/* file: quantiles_fpt.cpp */
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
//  Implementation of quantiles algorithm and types methods.
//--
*/

#include "quantiles_types.h"

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace interface1
{
/**
 * Allocates memory to store final results of the quantile algorithms
 * \param[in] input     Input objects for the quantiles algorithm
 * \param[in] parameter Parameters of the quantiles algorithm
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    const Parameter *par = static_cast<const Parameter *>(parameter);

    size_t nFeatures = in->get(data)->getNumberOfColumns();
    size_t nQuantileOrders = par->quantileOrders->getNumberOfColumns();

    Argument::set(quantiles, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nQuantileOrders, nFeatures,
                                                                                data_management::NumericTable::doAllocate)));
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace quantiles
}// namespace algorithms
}// namespace daal
