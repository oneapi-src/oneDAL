/* file: svm_train_cache.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_CACHE_H__
#define __SVM_TRAIN_CACHE_H__

#include "service/kernel/service_utils.h"
#include "externals/service_memory.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

/**
 * Types of caches for kernel function values
 */
enum SVMCacheType
{
    noCache,     /*!< No storage for caching kernel function values is provided */
    simpleCache, /*!< Storage for caching ALL kernel function values is provided */
    lruCache     /*!< Storage for caching PART of kernel function values is provided;
                         LRU algorithm is used to exclude values from cache */
};

/**
 * Common interface for cache that stores kernel function values
 */

template <typename algorithmFPType, CpuType cpu>
class SVMCacheCommonIface
{
public:
    virtual size_t getDataRowIndex(size_t rowIndex) const = 0;
};

template <Method method, typename algorithmFPType, CpuType cpu>
class SVMCacheIface : public SVMCacheCommonIface<algorithmFPType, cpu>
{};

template <Method method, typename algorithmFPType, CpuType cpu>
using SVMCachePtr = services::SharedPtr<SVMCacheIface<method, algorithmFPType, cpu> >;

template <Method method, SVMCacheType cacheType, typename algorithmFPType, CpuType cpu>
class SVMCache
{};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
