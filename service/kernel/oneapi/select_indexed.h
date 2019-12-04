/* file: select_indexed.h */
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

#ifndef __SELECT_INDEXED_H__
#define __SELECT_INDEXED_H__

#include "math_service_types.h"
#include "services/buffer.h"
#include "cl_kernels/select_indexed.cl"
#include "service_defines_oneapi.h"
#include "oneapi/internal/types_utils.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace selection
{

class QuickSelectIndexed
{
public:
    struct Result
    {
        UniversalBuffer values;
        UniversalBuffer indices;

        Result(ExecutionContextIface& context, uint32_t K,  uint32_t nVectors, TypeId valueType, TypeId indexType, services::Status* status)
            : values(context.allocate(valueType, nVectors * K, status)),
              indices(context.allocate(indexType, nVectors * K, status))
        {}
    };

public:
    static Result select(const UniversalBuffer& dataVectors, const UniversalBuffer& indexVectors,
                        const UniversalBuffer& rndSeq, uint32_t nRndSeq,
                        uint32_t K, uint32_t nVectors, uint32_t vectorSize,
                        uint32_t vectorOffset, services::Status* status);
    static Result& select(const UniversalBuffer& dataVectors, const UniversalBuffer& indexVectors,
                        const UniversalBuffer& rndSeq, uint32_t nRndSeq,
                        uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t vectorOffset,
                        Result& result, services::Status* status);
private:
    QuickSelectIndexed();
};

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
