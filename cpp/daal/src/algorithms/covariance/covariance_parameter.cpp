/* file: covariance_parameter.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "algorithms/covariance/covariance_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
/** Default constructor */
Parameter::Parameter() : daal::algorithms::Parameter(), outputMatrixType(covarianceMatrix) {}

} //namespace covariance
} // namespace algorithms
} // namespace daal
