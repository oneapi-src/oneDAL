/* file: dbscan_parameter_types.cpp */
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
//  Implementation of DBSCAN classes.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "daal_defines.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{

/**
 *  Constructs parameters of the DBSCAN algorithm
 */
Parameter::Parameter() :
    epsilon(0.5), minObservations(5), memorySavingMode(false),
    resultsToCompute(0),
    blockIndex(0), nBlocks(1),
    leftBlocks(1), rightBlocks(1) {}

/**
 *  Constructs parameters of the DBSCAN algorithm
 *  \param[in] _epsilon         Radius of neighborhood
 *  \param[in] _minObservations Minimal number of observations in neighborhood of core observation
 */
Parameter::Parameter(double _epsilon, size_t _minObservations) :
    epsilon(_epsilon), minObservations(_minObservations), memorySavingMode(false),
    resultsToCompute(0),
    blockIndex(0), nBlocks(1),
    leftBlocks(1), rightBlocks(1) {}

/**
 *  Constructs parameters of the DBSCAN algorithm by copying another parameters of the DBSCAN algorithm
 *  \param[in] other    Parameters of the DBSCAN algorithm
 */
Parameter::Parameter(const Parameter &other) :
    epsilon(other.epsilon), minObservations(other.minObservations), memorySavingMode(other.memorySavingMode),
    resultsToCompute(other.resultsToCompute),
    blockIndex(other.blockIndex), nBlocks(other.nBlocks),
    leftBlocks(other.leftBlocks), rightBlocks(other.rightBlocks) {}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(epsilon >= 0, services::ErrorIncorrectParameter, services::ParameterName, epsilonStr());
    DAAL_CHECK_EX(minObservations > 0, services::ErrorIncorrectParameter, services::ParameterName, minObservationsStr());
    return services::Status();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithm
} // namespace daal
