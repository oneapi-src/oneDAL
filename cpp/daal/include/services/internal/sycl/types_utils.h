/* file: types_utils.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_H__

#include "services/internal/sycl/types.h"

/// \cond INTERNAL
namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace interface1
{
template <typename...>
struct Typelist
{};

typedef Typelist<daal::services::internal::sycl::int8_t, daal::services::internal::sycl::int16_t, daal::services::internal::sycl::int32_t,
                 daal::services::internal::sycl::int64_t, daal::services::internal::sycl::uint8_t, daal::services::internal::sycl::uint16_t,
                 daal::services::internal::sycl::uint32_t, daal::services::internal::sycl::uint64_t, daal::services::internal::sycl::float32_t,
                 daal::services::internal::sycl::float64_t>
    PrimitiveTypes;

typedef Typelist<daal::services::internal::sycl::float32_t, daal::services::internal::sycl::float64_t> FloatTypes;

class TypeDispatcher
{
public:
    template <typename Operation>
    static void dispatch(TypeId type, Operation && op, Status & status)
    {
        dispatchInternal(status, type, op, PrimitiveTypes());
    }

    template <typename Operation>
    static void floatDispatch(TypeId type, Operation && op, Status & status)
    {
        dispatchInternal(status, type, op, FloatTypes());
    }

private:
    template <typename Operation, typename Head, typename... Rest>
    static void dispatchInternal(Status & status, TypeId type, Operation && op, Typelist<Head, Rest...>)
    {
        if (type == TypeIds::id<Head>())
        {
            op(Typelist<Head>(), status);
        }
        else
        {
            dispatchInternal(status, type, op, Typelist<Rest...>());
        }
    }

    template <typename Operation>
    static void dispatchInternal(Status & status, TypeId type, Operation && op, Typelist<>)
    {
        DAAL_ASSERT(!"Unknown type");
    }
};

String getKeyFPType(TypeId typeId);

} // namespace interface1

using interface1::Typelist;
using interface1::TypeDispatcher;
using interface1::getKeyFPType;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
