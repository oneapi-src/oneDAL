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

#pragma once

#include "oneapi/dal/data/detail/table_reader_impl.hpp"
#include "oneapi/dal/data/table.hpp"

namespace oneapi::dal {

template <typename T>
struct is_csv_table_reader_impl {
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(table, read, (const char * file_name))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void, set_delimiter, (char delimiter))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void, set_parse_header, (bool parse_header))

    static constexpr bool value_host = has_method_read_v<T>
        && has_method_set_delimiter_v<T>
        && has_method_set_parse_header_v<T>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    ONEAPI_DAL_HAS_METHOD_TRAIT(table, read, (sycl::queue & queue, const char * file_name), read_dpc)

    static constexpr bool value_dpc = has_method_read_dpc_v<T>;
    static constexpr bool value     = value_host && value_dpc;
#else
    static constexpr bool value = value_host;
#endif
};

template <typename T>
inline constexpr bool is_csv_table_reader_impl_v = is_csv_table_reader_impl<T>::value;

class ONEAPI_DAL_EXPORT csv_table_reader {
    friend detail::pimpl_accessor;
    using pimpl_t = detail::pimpl<detail::csv_table_reader_impl_iface>;

public:
    csv_table_reader();

    template <typename Impl,
              typename ImplType = std::decay_t<Impl>,
              typename          = std::enable_if_t<is_csv_table_reader_impl_v<ImplType> &&
                                          !std::is_base_of_v<csv_table_reader, ImplType>>>
    csv_table_reader(Impl&& impl)
            : csv_table_reader(new detail::csv_table_reader_impl_wrapper(std::forward<Impl>(impl))) {}

    table read(const char * file_name) {
        auto& impl = get_impl();
        return impl.read(file_name);
    }

    auto& set_delimiter(char delimiter) {
        auto& impl = get_impl();
        impl.set_delimiter(delimiter);
        return *this;
    }

    auto& set_parse_header(bool parse_header) {
        auto& impl = get_impl();
        impl.set_parse_header(parse_header);
        return *this;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    table read(sycl::queue& queue,
               const char * file_name) {
        auto& impl = get_impl();
        return impl.read(queue, file_name);
    }
#endif

protected:
    csv_table_reader(detail::csv_table_reader_impl_iface* obj) : impl_(obj) {}

private:
    detail::csv_table_reader_impl_iface& get_impl() {
        return detail::get_impl<detail::csv_table_reader_impl_iface>(*this);
    }

    pimpl_t impl_;
};

} // namespace oneapi::dal
