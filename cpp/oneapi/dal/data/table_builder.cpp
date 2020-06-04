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

#include "oneapi/dal/data/backend/homogen_table_impl.hpp"
#include "oneapi/dal/data/table_builder.hpp"

namespace dal {
namespace backend {

class table_builder_impl {
    using dense_rw_storage = detail::table_builder_impl_iface::dense_rw_storage;

public:
    table_builder_impl(table&& t)
        : table_impl_(detail::pimpl_accessor().get_pimpl(std::move(t))) {}

    auto build_table() {
        return detail::pimpl_accessor().make_from_pimpl<table>(table_impl_);
    }

    auto get_storage() -> dense_rw_storage& {
        return *table_impl_;
    }

private:
    detail::pimpl<detail::table_impl_iface> table_impl_;
};

class homogen_table_builder_impl {
public:
    using dense_rw_storage = detail::table_builder_impl_iface::dense_rw_storage;
    using table_impl_t = detail::homogen_table_impl_iface;
    using pimpl_t = detail::pimpl<table_impl_t>;

public:
    homogen_table_builder_impl(homogen_table&& t) {
        auto& base_pimpl = detail::pimpl_accessor().get_pimpl(std::move(t));
        table_impl_ = std::static_pointer_cast<table_impl_t>(base_pimpl);
    }

    homogen_table_builder_impl(const pimpl_t& table_impl)
        : table_impl_(table_impl) {}

    table build_table() {
        return build_homogen_table();
    }

    homogen_table build_homogen_table() {
        return detail::pimpl_accessor().make_from_pimpl<homogen_table>(table_impl_);
    }

    dense_rw_storage& get_storage() {
        return *table_impl_;
    }

private:
    pimpl_t table_impl_;
};

} // namespace backend

table_builder::table_builder(table&& t)
    : table_builder(backend::table_builder_impl{ std::move(t) }) {}

template <typename DataType>
homogen_table_builder::homogen_table_builder(std::int64_t row_count, std::int64_t column_count,
                                             const DataType* data_pointer,
                                             data_layout layout)
    : homogen_table_builder(homogen_table{ row_count, column_count, data_pointer, layout }) {}

template <typename DataType, typename>
homogen_table_builder::homogen_table_builder(std::int64_t row_count, std::int64_t column_count,
                                             DataType value,
                                             data_layout layout)
    : table_builder(backend::homogen_table_builder_impl {
        backend::homogen_table_builder_impl::pimpl_t {
            new detail::homogen_table_impl_wrapper { backend::homogen_table_impl{ row_count, column_count, value, layout } }
        }
    }) {}

template <typename DataType>
homogen_table_builder::homogen_table_builder(std::int64_t column_count,
                                             const array<DataType>& data,
                                             data_layout layout)
    : table_builder(backend::homogen_table_builder_impl {
        backend::homogen_table_builder_impl::pimpl_t {
            new detail::homogen_table_impl_wrapper { backend::homogen_table_impl{ column_count, data, layout } }
        }
    }) {}

homogen_table_builder::homogen_table_builder(homogen_table&& t)
    : table_builder(backend::homogen_table_builder_impl{ std::move(t) }) {}

homogen_table homogen_table_builder::build() const {
    using impl_t = backend::homogen_table_builder_impl;
    using wrapper_t = detail::table_builder_impl_wrapper<impl_t>;

    auto& impl = detail::get_impl<wrapper_t>(*this).get();
    return impl.build_homogen_table();
}

template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, const float*, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, const double*, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, const std::int32_t*, data_layout);

template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, float, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, double, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, std::int64_t, std::int32_t, data_layout);

template homogen_table_builder::homogen_table_builder(std::int64_t, const array<float>&, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, const array<double>&, data_layout);
template homogen_table_builder::homogen_table_builder(std::int64_t, const array<std::int32_t>&, data_layout);

} // namespace dal
