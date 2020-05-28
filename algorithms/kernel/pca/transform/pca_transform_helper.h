/* file: pca_transform_helper.h */
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

#ifndef __PCA_TRANSFORM_HELPER_H__
#define __PCA_TRANSFORM_HELPER_H__

#include "service/kernel/service_environment.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct BSHelper
{
    static size_t getBlockSize(const size_t m, const size_t k, const size_t n)
    {
        const double cacheFullness     = 0.8;
        const size_t maxRowsPerBlock   = 512;
        const size_t minRowsPerBlockL1 = 256;
        const size_t minRowsPerBlockL2 = 8;
        size_t rowsFitL1               = (services::internal::getL1CacheSize() / sizeof(algorithmFPType) * cacheFullness - m * k) / (k + m);
        size_t rowsFitL2               = (services::internal::getL2CacheSize() / sizeof(algorithmFPType) * cacheFullness - m * k) / (k + m);
        size_t blockSize               = 96;

        if (rowsFitL1 >= minRowsPerBlockL1 && rowsFitL1 <= maxRowsPerBlock)
        {
            blockSize = rowsFitL1;
        }
        else if (rowsFitL2 >= minRowsPerBlockL2 && rowsFitL2 <= maxRowsPerBlock)
        {
            blockSize = rowsFitL2;
        }
        else if (rowsFitL2 >= maxRowsPerBlock)
        {
            blockSize = maxRowsPerBlock;
        }
        return blockSize;
    }
};
} // namespace internal
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
