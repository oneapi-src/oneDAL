/* file: dbscan_parameter_types.cpp */
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
//  Implementation of DBSCAN classes.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "services/daal_defines.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
/**
 *  Constructs parameters of the DBSCAN algorithm
 */
Parameter::Parameter()
    : epsilon(0.5), minObservations(5), memorySavingMode(false), resultsToCompute(0), blockIndex(0), nBlocks(1), leftBlocks(1), rightBlocks(1)
{}

/**
 *  Constructs parameters of the DBSCAN algorithm
 *  \param[in] _epsilon         Radius of neighborhood
 *  \param[in] _minObservations Minimal number of observations in neighborhood of core observation
 */
Parameter::Parameter(double _epsilon, size_t _minObservations)
    : epsilon(_epsilon),
      minObservations(_minObservations),
      memorySavingMode(false),
      resultsToCompute(0),
      blockIndex(0),
      nBlocks(1),
      leftBlocks(1),
      rightBlocks(1)
{}

/**
 *  Constructs parameters of the DBSCAN algorithm by copying another parameters of the DBSCAN algorithm
 *  \param[in] other    Parameters of the DBSCAN algorithm
 */
Parameter::Parameter(const Parameter & other)
    : epsilon(other.epsilon),
      minObservations(other.minObservations),
      memorySavingMode(other.memorySavingMode),
      resultsToCompute(other.resultsToCompute),
      blockIndex(other.blockIndex),
      nBlocks(other.nBlocks),
      leftBlocks(other.leftBlocks),
      rightBlocks(other.rightBlocks)
{}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(epsilon >= 0, services::ErrorIncorrectParameter, services::ParameterName, epsilonStr());
    DAAL_CHECK_EX(minObservations > 0, services::ErrorIncorrectParameter, services::ParameterName, minObservationsStr());
    return services::Status();
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal
