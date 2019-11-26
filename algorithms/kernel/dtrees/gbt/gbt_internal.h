/* file: gbt_internal.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Internal definitions of gradient boosted trees algorithm
//--
*/

#ifndef __GBT_INTERNAL__
#define __GBT_INTERNAL__

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{
/**
* Parallel options in GBT training algorithm
*/
enum ParallelOptions
{
    parallelFeatures = 1, /*!< Process features in parallel threads when finding splits */
    parallelNodes    = 2, /*!< Process nodes in parallel threads when building a tree */
    parallelTrees    = 4, /*!< Multi-class classification only: process single class tree in a separate thread */
    parallelAll      = (parallelFeatures | parallelNodes | parallelTrees)
};

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
