/* file: types_utils.cpp */
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

#include "services/internal/sycl/types_utils.h"

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
struct TypeToStringConverter
{
    services::String result;

    template <typename T>
    void operator()(Typelist<T>, Status & status)
    {
        result = daal::services::internal::sycl::getKeyFPType<T>();
    }
};

services::String getKeyFPType(TypeId typeId)
{
    Status status;

    TypeToStringConverter converter;
    TypeDispatcher::dispatch(typeId, converter, status);

    return converter.result;
}

} // namespace interface1
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
