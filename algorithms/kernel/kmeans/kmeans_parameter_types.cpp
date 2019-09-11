/* file: kmeans_parameter_types.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_types.h"
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

namespace interface1
{

/**
 *  Constructs parameters of the K-Means algorithm
 *  \param[in] _nClusters   Number of clusters
 *  \param[in] _maxIterations Number of iterations
 */
Parameter::Parameter(size_t _nClusters, size_t _maxIterations) :
    nClusters(_nClusters), maxIterations(_maxIterations), accuracyThreshold(0.0), gamma(1.0),
    distanceType(euclidean), assignFlag(true) {}

/**
 *  Constructs parameters of the K-Means algorithm by copying another parameters of the K-Means algorithm
 *  \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter &other) :
    nClusters(other.nClusters), maxIterations(other.maxIterations),
    accuracyThreshold(other.accuracyThreshold), gamma(other.gamma),
    distanceType(other.distanceType), assignFlag(other.assignFlag)
{}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nClusters > 0, ErrorIncorrectParameter, ParameterName, nClustersStr());
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(gamma >= 0, ErrorIncorrectParameter, ParameterName, gammaStr());
    return services::Status();
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal
