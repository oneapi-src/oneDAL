/* file: kmeans_init_impl.h */
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
//  Implementation of kmeans init classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
enum LocalData
{
    numberOfClusters,       //Number of clusters (candidates) selected so far
    closestClusterDistance, //Distance from every row in the input data to its closest cluster (candidate)
    closestCluster,         //parallelPlus only. Index of the closest cluster (candidate) for every row in the input data
    candidateRating,        //parallelPlus only. For each candidate number of closest points among input data
    localDataSize
};

#define isParallelPlusMethod(method) ((method == kmeans::init::parallelPlusDense) || (method == kmeans::init::parallelPlusCSR))

services::Status checkLocalData(const data_management::DataCollection * pInput, const Parameter * par, const char * dataName,
                                const data_management::NumericTable * pData, bool bParallelPlus);

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
