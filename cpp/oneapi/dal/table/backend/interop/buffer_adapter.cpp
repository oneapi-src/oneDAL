/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/interop/error_converter.hpp"

#include "oneapi/dal/table/backend/interop/buffer_adapter.hpp"

namespace oneapi::dal::backend::interop {

template <typename DataType>
struct array_deleter {
    using data_t = DataType;

    array_deleter() = delete;
    array_deleter(array_deleter&&) = default;
    array_deleter(const array_deleter&) = default;

    array_deleter(std::shared_ptr<data_t> inp) : ptr(std::move(inp)) {}

    void operator()(const void* del_ptr) {
        return ptr.reset();
    }

    std::shared_ptr<data_t> ptr;
};

template <typename DataType>
auto convert(std::shared_ptr<DataType> inp) {
    using daal::services::SharedPtr;
    using data_t = std::remove_const_t<DataType>;

    auto del = array_deleter(inp);
    DataType* const raw_ptr = inp.get();
    return SharedPtr<data_t>(raw_ptr, del);
}

template <typename DataType>
auto to_daal_shared(const detail::array_impl<DataType>& arr_impl) {
    using daal::services::SharedPtr;

    SharedPtr<DataType> curr_ptr;
    if (arr_impl.has_mutable_data()) {
        auto raw = arr_impl.get_shared();
        curr_ptr = convert(std::move(raw));
    }
    else if (arr_impl.has_data()) {
        auto raw = arr_impl.get_cshared();
        curr_ptr = convert(std::move(raw));
    }
    else {
        ONEDAL_ASSERT(!"An empty array");
    }

    return curr_ptr;
}

template <typename DataType>
auto convert_with_status(const dal::array<DataType>& array)
    -> std::pair<buffer_t<DataType>, daal::services::Status> {
    using dal::detail::integral_cast;
    using daal::services::internal::Buffer;

    const auto count = array.get_count();
    auto& arr_impl = detail::get_impl(array);
    ONEDAL_ASSERT(count == arr_impl.get_count());
    auto ccount = integral_cast<std::size_t>(count);

    daal::services::Status st;
    auto shared = to_daal_shared(arr_impl);
    auto buf = Buffer<DataType>(shared, ccount, st);

    return std::make_pair(buf, st);
}

template <typename DataType>
auto convert(const dal::array<DataType>& array) -> buffer_t<DataType> {
    auto [buffer, status] = convert_with_status(array);
    interop::status_to_exception(status);
    return buffer;
}

#define INSTANTIATE(TYPE)                                           \
    template auto convert(const dal::array<TYPE>&)->buffer_t<TYPE>; \
    template auto convert_with_status(const dal::array<TYPE>&)      \
        ->std::pair<buffer_t<TYPE>, daal::services::Status>;

INSTANTIATE(int)
INSTANTIATE(long)
INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::interop
