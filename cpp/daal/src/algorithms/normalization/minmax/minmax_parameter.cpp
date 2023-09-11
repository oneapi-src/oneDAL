/* file: minmax_parameter.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "algorithms/normalization/minmax_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
/** Constructs min-max normalization parameters */
DAAL_EXPORT ParameterBase::ParameterBase(double lowerBound, double upperBound, const SharedPtr<low_order_moments::BatchImpl> & moments)
    : lowerBound(lowerBound), upperBound(upperBound), moments(moments)
{}

/**
 * Check the correctness of the %ParameterBase object
 */
DAAL_EXPORT Status ParameterBase::check() const
{
    DAAL_CHECK(moments, ErrorNullParameterNotSupported);
    DAAL_CHECK(lowerBound < upperBound, ErrorLowerBoundGreaterThanOrEqualToUpperBound);
    return Status();
}

} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
