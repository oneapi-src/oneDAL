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

#include <type_traits>

#include "oneapi/dal/detail/common_dpc.hpp"
#include "oneapi/dal/table/detail/table_impl_wrapper.hpp"
#include "oneapi/dal/util/type_traits.hpp"

namespace oneapi::dal {

class ONEAPI_DAL_EXPORT table {
    friend detail::pimpl_accessor;
    using pimpl = detail::pimpl<detail::table_impl_iface>;

public:
    table();
    table(const table&) = default;
    table(table&&);

    template <typename Impl,
              typename ImplType = std::decay_t<Impl>,
              typename          = std::enable_if_t<is_table_impl_v<ImplType> &&
                                          !std::is_base_of_v<table, ImplType>>>
    table(Impl&& impl) {
        init_impl(new detail::table_impl_wrapper(std::forward<Impl>(impl)));
    }

    table& operator=(const table&) = default;
    table& operator                =(table&&);

    bool has_data() const noexcept;
    std::int64_t get_column_count() const;
    std::int64_t get_row_count() const;
    const table_metadata& get_metadata() const;
    std::int64_t get_kind() const;

protected:
    table(const pimpl& impl) : impl_(impl) {}

    void init_impl(pimpl::element_type* impl);

private:
    pimpl impl_;
};

} // namespace oneapi::dal
