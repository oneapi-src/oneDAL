/* file: sorter.h */
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

#ifndef __SORTER_H__
#define __SORTER_H__

#include "src/sycl/math_service_types.h"
#include "services/internal/buffer.h"
#include "src/sycl/cl_kernels/radix_sort.cl"
#include "services/internal/sycl/types_utils.h"
#include "services/internal/sycl/execution_context.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace sort
{
class RadixSort
{
public:
    RadixSort() = delete;

    static services::Status sort(const UniversalBuffer & input, const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors,
                                 uint32_t vectorSize, uint32_t vectorOffset);

    static services::Status sortIndices(UniversalBuffer & values, UniversalBuffer & indices, UniversalBuffer & valuesOut,
                                        UniversalBuffer & indicesOut, uint32_t nRows);

    static services::Status radixScan(UniversalBuffer & values, UniversalBuffer & partialHists, uint32_t nRows, uint32_t bitOffset,
                                      uint32_t localSize, uint32_t nLocalHists);

    static services::Status radixHistScan(UniversalBuffer & values, UniversalBuffer & partialHists, UniversalBuffer & partialPrefixHists,
                                          uint32_t localSize, uint32_t nSubgroupHists);

    static services::Status radixReorder(UniversalBuffer & valuesSrc, UniversalBuffer & indicesSrc, UniversalBuffer & partialPrefixHists,
                                         UniversalBuffer & valuesDst, UniversalBuffer & indicesDst, uint32_t nRows, uint32_t bitOffset,
                                         uint32_t localSize, uint32_t nLocalHists);

private:
    static const uint32_t _preferableSubGroup = 16; // preferable maximal sub-group size
    static const uint32_t _radixBits          = 4;  // number of bits used for a single pass of radix sort
};

} // namespace sort
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
