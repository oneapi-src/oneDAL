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

#pragma once

#include <type_traits>

#include "oneapi/dal/data/detail/table_impl_wrapper.hpp"
#include "oneapi/dal/util/type_traits.hpp"

namespace dal {

class table {
    friend detail::pimpl_accessor;
    using pimpl = detail::pimpl<detail::table_impl_iface>;

public:
    table();
    table(const table&) = default;
    table(table&&);

    template <typename Impl,
              typename = std::enable_if_t<is_table_impl_v<std::decay_t<Impl>>>>
    table(Impl&& impl) {
        init_impl(new detail::table_impl_wrapper(std::forward<Impl>(impl)));
    }

    table& operator=(const table&) = default;
    table& operator=(table&&);

    bool has_data() const noexcept;
    std::int64_t get_column_count() const;
    std::int64_t get_row_count() const;
    const table_metadata& get_metadata() const;

protected:
    table(const pimpl& impl)
        : impl_(impl) {}

    void init_impl(pimpl::element_type* impl);

private:
    pimpl impl_;
};

class homogen_table : public table {
    friend detail::pimpl_accessor;
    using pimpl = detail::pimpl<detail::homogen_table_impl_iface>;

public:
    homogen_table() = default;

    template <typename Impl,
              typename = std::enable_if_t<is_homogen_table_impl_v<std::decay_t<Impl>>>>
    homogen_table(Impl&& impl) {
        // TODO: usage of protected method of base class: a point to break inheritance?
        init_impl(new detail::homogen_table_impl_wrapper(std::forward<Impl>(impl)));
    }

    template <typename DataType>
    homogen_table(std::int64_t row_count, std::int64_t column_count,
                  const DataType* data_pointer,
                  data_layout layout = data_layout::row_major);

    template <typename DataType>
    const DataType* get_data() const {
        using impl_t = detail::homogen_table_impl_iface;

        auto& impl = detail::get_impl<impl_t>(*this);
        return reinterpret_cast<const DataType*>(impl.get_data());
    }

private:
    homogen_table(const pimpl& impl)
        : table(impl) {}
};

} // namespace dal
