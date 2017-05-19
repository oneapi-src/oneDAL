/* file: kmeans_init_parameter_types.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{

/**
 *  Main constructor
 *  \param[in] _nClusters     Number of clusters
 *  \param[in] _offset        Offset in the total data set specifying the start of a block stored on a given local node
 *  \param[in] seed           Seed for generating random numbers for the initialization
 */
Parameter::Parameter(size_t _nClusters, size_t _offset, size_t seed) : nClusters(_nClusters), offset(_offset), nRowsTotal(0), seed(seed),
    oversamplingFactor(0.5), nRounds(5) {}

/**
 * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
 * by copying another parameters object
 * \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter &other) : nClusters(other.nClusters), offset(other.offset), nRowsTotal(other.nRowsTotal), seed(other.seed),
    oversamplingFactor(other.oversamplingFactor), nRounds(other.nRounds) {}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nClusters > 0, ErrorIncorrectParameter, ParameterName, nClustersStr());
    return services::Status();
}

DistributedStep2LocalPlusPlusParameter::DistributedStep2LocalPlusPlusParameter(size_t _nClusters, bool bFirstIteration) :
    Parameter(_nClusters), outputForStep5Required(false), firstIteration(bFirstIteration)
{}

DistributedStep2LocalPlusPlusParameter::DistributedStep2LocalPlusPlusParameter(const DistributedStep2LocalPlusPlusParameter &other) :
    Parameter(other), outputForStep5Required(other.outputForStep5Required), firstIteration(other.firstIteration)
{}

services::Status DistributedStep2LocalPlusPlusParameter::check() const
{
    return services::Status();
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
