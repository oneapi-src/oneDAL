/* file: regression_training_partialresult.cpp */
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
//  Implementation of the class defining the partial result of the regression training algorithm
//--
*/

#include "services/daal_defines.h"
#include "algorithms/regression/regression_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
PartialResult::PartialResult(size_t nElements) : daal::algorithms::PartialResult(nElements) {}
} // namespace training
} // namespace regression
} // namespace algorithms
} // namespace daal
