/* file: partition.h */
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

#ifndef __PARTITION_H__
#define __PARTITION_H__

#include "service/kernel/oneapi/math_service_types.h"
#include "services/buffer.h"
#include "oneapi/internal/types_utils.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
class Partition
{
public:
    Partition() = delete;

    static services::Status flaggedIndex(UniversalBuffer mask, UniversalBuffer data, UniversalBuffer outData, const size_t nElems, size_t & nSelect);
    static services::Status flagged(UniversalBuffer mask, UniversalBuffer data, UniversalBuffer outData, const size_t nElems, size_t & nSelect);

protected:
    static services::Status reorderIndex(ClKernelFactoryIface & kernelFactory, UniversalBuffer & mask, UniversalBuffer & data,
                                         UniversalBuffer & outData, UniversalBuffer & partialPrefixSums, size_t nElems, size_t localSize,
                                         size_t nLocalSums);

    static services::Status reorder(ClKernelFactoryIface & kernelFactory, UniversalBuffer & mask, UniversalBuffer & data, UniversalBuffer & outData,
                                    UniversalBuffer & partialPrefixSums, size_t nElems, size_t localSize, size_t nLocalSums);

    static services::Status scanIndex(ClKernelFactoryIface & factory, UniversalBuffer & mask, UniversalBuffer & data, UniversalBuffer & partialSums,
                                      size_t nElems, size_t localSize, size_t nLocalSums);

    static services::Status sumScan(ClKernelFactoryIface & kernelFactory, UniversalBuffer & partialSums, UniversalBuffer & partialPrefixSums,
                                    UniversalBuffer & totalSum, size_t localSize, size_t nSubgroupSums);

    static services::Status scan(ClKernelFactoryIface & kernelFactory, UniversalBuffer & mask, UniversalBuffer & partialSums, size_t nElems,
                                 size_t localSize, size_t nLocalSums);

private:
    static services::Status buildProgram(ClKernelFactoryIface & factory, const TypeId & vectorTypeId);

private:
    static const uint32_t _preferableSubGroup = 16; // preferable maximal sub-group size
    static const uint32_t _maxLocalSums       = 256;
};

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
