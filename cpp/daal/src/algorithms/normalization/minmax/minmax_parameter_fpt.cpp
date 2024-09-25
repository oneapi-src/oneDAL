/* file: minmax_parameter_fpt.cpp */
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
namespace interface1
{
typedef SharedPtr<low_order_moments::BatchImpl> LowOrderMomentsPtr;

/** Constructs min-max normalization parameters with default low order algorithm */
template <typename algorithmFPType>
Parameter<algorithmFPType>::Parameter(double lowerBound, double upperBound)
    : ParameterBase(lowerBound, upperBound, LowOrderMomentsPtr(new low_order_moments::Batch<algorithmFPType>()))
{}

/** Constructs min-max normalization parameters */
template <typename algorithmFPType>
Parameter<algorithmFPType>::Parameter(double lowerBound, double upperBound, const LowOrderMomentsPtr & moments)
    : ParameterBase(lowerBound, upperBound, moments)
{}

template Parameter<DAAL_FPTYPE>::Parameter(double lowerBound, double upperBound);

template Parameter<DAAL_FPTYPE>::Parameter(double lowerBound, double upperBound, const LowOrderMomentsPtr & moments);

} // namespace interface1
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
