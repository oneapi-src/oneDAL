/* file: kmeans_types_v1.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of K-Means algorithm interface.
//--
*/

#ifndef __KMEANS_TYPES_V1_H__
#define __KMEANS_TYPES_V1_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Constructs parameters of K-Means algorithm
     *  \param[in] _nClusters   Number of clusters
     *  \param[in] _maxIterations Number of iterations
     */
    Parameter(size_t _nClusters, size_t _maxIterations);

    /**
     *  Constructs parameters of K-Means algorithm by copying another parameters of K-Means algorithm
     *  \param[in] other    Parameters of K-Means algorithm
     */
    Parameter(const Parameter & other);

    size_t nClusters;          /*!< Number of clusters */
    size_t maxIterations;      /*!< Number of iterations */
    double accuracyThreshold;  /*!< Threshold for the termination of the algorithm */
    double gamma;              /*!< Weight used in distance computation for categorical features */
    DistanceType distanceType; /*!< Distance used in the algorithm */
    bool assignFlag;           /*!< Do data points assignment */

    services::Status check() const DAAL_C11_OVERRIDE;
};
} // namespace interface1
} // namespace kmeans
} // namespace algorithms
} // namespace daal
#endif
