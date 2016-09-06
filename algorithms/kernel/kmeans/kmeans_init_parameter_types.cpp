/* file: kmeans_init_parameter_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"

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
Parameter::Parameter(size_t _nClusters, size_t _offset, size_t seed) : nClusters(_nClusters), offset(_offset), nRowsTotal(0), seed(seed) {}

/**
 * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
 * by copying another parameters object
 * \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter &other) : nClusters(other.nClusters), offset(other.offset), nRowsTotal(other.nRowsTotal), seed(other.seed) {}

void Parameter::check() const
{
    DAAL_CHECK_EX(nClusters > 0, ErrorIncorrectParameter, ParameterName, nClustersStr());
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
