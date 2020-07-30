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

#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/backend/empty_table_impl.hpp"
#include "oneapi/dal/data/backend/homogen_table_impl.hpp"

using std::int64_t;

namespace oneapi::dal {

table::table() : table(backend::empty_table_impl{}) {}

table::table(table&& t) : impl_(std::move(t.impl_)) {
    using wrapper     = detail::table_impl_wrapper<backend::empty_table_impl>;
    using wrapper_ptr = detail::shared<wrapper>;

    t.impl_ = wrapper_ptr(new wrapper(backend::empty_table_impl{}));
}

table& table::operator=(table&& t) {
    this->impl_.swap(t.impl_);
    return *this;
}

bool table::has_data() const noexcept {
    return impl_->get_column_count() > 0 && impl_->get_row_count() > 0;
}

int64_t table::get_column_count() const {
    return impl_->get_column_count();
}

int64_t table::get_row_count() const {
    return impl_->get_row_count();
}

const table_metadata& table::get_metadata() const {
    return impl_->get_metadata();
}

int64_t table::get_kind() const {
    return impl_->get_kind();
}

void table::init_impl(detail::table_impl_iface* impl) {
    impl_ = pimpl{ impl };
}

int64_t homogen_table::kind() {
    return 1;
}

homogen_table::homogen_table() : homogen_table(backend::homogen_table_impl{}) {}

const homogen_table_metadata& homogen_table::get_metadata() const {
    const auto& impl = detail::get_impl<detail::homogen_table_impl_iface>(*this);
    return impl.get_metadata();
}

const void* homogen_table::get_data() const {
    const auto& impl = detail::get_impl<detail::homogen_table_impl_iface>(*this);
    return impl.get_data();
}

template <typename Policy>
void homogen_table::init_impl(const Policy& policy,
                              int64_t row_count,
                              int64_t column_count,
                              const array<byte_t>& data,
                              const table_feature& feature,
                              homogen_data_layout layout) {
    init_impl(backend::homogen_table_impl(column_count, data, feature, layout));
}

template ONEAPI_DAL_EXPORT void homogen_table::init_impl(const detail::cpu_dispatch_default&,
                                                         int64_t,
                                                         int64_t,
                                                         const array<byte_t>&,
                                                         const table_feature&,
                                                         homogen_data_layout);

#ifdef ONEAPI_DAL_DATA_PARALLEL
template ONEAPI_DAL_EXPORT void homogen_table::init_impl(const data_parallel_policy&,
                                                         int64_t,
                                                         int64_t,
                                                         const array<byte_t>&,
                                                         const table_feature&,
                                                         homogen_data_layout);
#endif

} // namespace oneapi::dal
