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

void table::init_impl(detail::table_impl_iface* impl) {
    impl_ = pimpl{ impl };
}

template <typename DataType>
homogen_table::homogen_table(int64_t row_count,
                             int64_t column_count,
                             const DataType* data_pointer,
                             data_layout layout)
        : homogen_table(
              backend::homogen_table_impl(row_count, column_count, data_pointer, layout)) {}

template ONEAPI_DAL_EXPORT homogen_table::homogen_table(int64_t,
                                                        int64_t,
                                                        const float*,
                                                        data_layout);
template ONEAPI_DAL_EXPORT homogen_table::homogen_table(int64_t,
                                                        int64_t,
                                                        const double*,
                                                        data_layout);
template ONEAPI_DAL_EXPORT homogen_table::homogen_table(int64_t,
                                                        int64_t,
                                                        const std::int32_t*,
                                                        data_layout);

} // namespace oneapi::dal
