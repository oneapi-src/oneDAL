/* file: kmeans_init_parameter_types.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

Parameter::Parameter(size_t _nClusters, size_t _offset, size_t _seed) : nClusters(_nClusters), offset(_offset), nRowsTotal(0), seed(_seed),
    oversamplingFactor(0.5), nRounds(5), engine(engines::mt19937::Batch<>::create(_seed)) {}

/**
 * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
 * by copying another parameters object
 * \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter &other) : nClusters(other.nClusters), offset(other.offset), nRowsTotal(other.nRowsTotal), seed(other.seed),
    oversamplingFactor(other.oversamplingFactor), nRounds(other.nRounds), engine(other.engine) {}

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
