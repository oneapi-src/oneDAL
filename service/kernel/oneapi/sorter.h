/* file: sorter.h */
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

#ifndef __SORTER_H__
#define __SORTER_H__

#include "service/kernel/oneapi/math_service_types.h"
#include "services/buffer.h"
#include "service/kernel/oneapi/cl_kernels/radix_sort.cl"
#include "oneapi/internal/types_utils.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace sort
{
class RadixSort
{
public:
    static void sort(const UniversalBuffer & input, const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors,
                     uint32_t vectorSize, uint32_t vectorOffset, services::Status * status);


    static services::Status sortIndeces(const UniversalBuffer & values, const UniversalBuffer & indices, const UniversalBuffer & valuesOut,
                                                                   const UniversalBuffer & indicesOut, int nRows);

    static services::Status radixScan(const UniversalBuffer & values, UniversalBuffer & partialHists, int nRows, int bitOffset,
                                                                   int localSize, int nLocalHists);

    static services::Status radixHistScan(const UniversalBuffer & values, UniversalBuffer & partialHists, UniversalBuffer & partialPrefixHists,
                                                                       int localSize, int nSubgroupHists);

    static services::Status radixReorder(const UniversalBuffer & valuesSrc, const UniversalBuffer & indicesSrc,
                                        UniversalBuffer & partialPrefixHists, const UniversalBuffer & valuesDst,
                                          const UniversalBuffer & indicesDst, int nRows, int bitOffset, int localSize,
                                          int nLocalHists);

private:
    RadixSort();
    static const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
    static const uint32_t _radixBits            = 4;    // preferable maximal sub-group size

};

} // namespace sort
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
