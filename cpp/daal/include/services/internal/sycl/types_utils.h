/* file: types_utils.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_H__

#include "services/internal/sycl/types.h"

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
/** @ingroup oneapi_internal
 * @{
 */

template <typename...>
struct Typelist
{};

typedef Typelist<daal::services::internal::sycl::int8_t, daal::services::internal::sycl::int16_t, daal::services::internal::sycl::int32_t,
                 daal::services::internal::sycl::int64_t, daal::services::internal::sycl::uint8_t, daal::services::internal::sycl::uint16_t,
                 daal::services::internal::sycl::uint32_t, daal::services::internal::sycl::uint64_t, daal::services::internal::sycl::float32_t,
                 daal::services::internal::sycl::float64_t>
    PrimitiveTypes;

typedef Typelist<daal::services::internal::sycl::float32_t, daal::services::internal::sycl::float64_t> FloatTypes;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__TYPEDISPATCHER"></a>
 *  \brief Makes runtime dispatching of types
 */
class TypeDispatcher
{
public:
    template <typename Operation>
    static void dispatch(TypeId type, Operation && op)
    {
        dispatchInternal(type, op, PrimitiveTypes());
    }

    template <typename Operation>
    static void floatDispatch(TypeId type, Operation && op)
    {
        dispatchInternal(type, op, FloatTypes());
    }

private:
    template <typename Operation, typename Head, typename... Rest>
    static void dispatchInternal(TypeId type, Operation && op, Typelist<Head, Rest...>)
    {
        if (type == TypeIds::id<Head>())
        {
            op(Typelist<Head>());
        }
        else
        {
            dispatchInternal(type, op, Typelist<Rest...>());
        }
    }

    template <typename Operation>
    static void dispatchInternal(TypeId type, Operation && op, Typelist<>)
    {
        DAAL_ASSERT(!"Unknown type");
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__TYPETOSTRINGCONVERTER"></a>
 *  \brief Converts type to string representation
 */
struct TypeToStringConverter
{
    services::String result;

    template <typename T>
    void operator()(Typelist<T>)
    {
        result = daal::services::internal::sycl::getKeyFPType<T>();
    }
};

services::String getKeyFPType(TypeId typeId);

/** @} */

} // namespace interface1

using interface1::Typelist;
using interface1::TypeDispatcher;
using interface1::getKeyFPType;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
