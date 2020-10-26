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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/backend/homogen_table_impl.hpp"

using std::int64_t;

namespace oneapi::dal {

int64_t homogen_table::kind() {
    return 1;
}

homogen_table::homogen_table() : homogen_table(backend::homogen_table_impl{}) {}

const void* homogen_table::get_data() const {
    const auto& impl = detail::get_impl<detail::homogen_table_impl_iface>(*this);
    return impl.get_data();
}

template <typename Policy>
void homogen_table::init_impl(const Policy& policy,
                              int64_t row_count,
                              int64_t column_count,
                              const array<byte_t>& data,
                              const data_type& dtype,
                              data_layout layout) {
    init_impl(backend::homogen_table_impl(column_count, data, dtype, layout));
}

template ONEDAL_EXPORT void homogen_table::init_impl(const detail::default_host_policy&,
                                                     int64_t,
                                                     int64_t,
                                                     const array<byte_t>&,
                                                     const data_type&,
                                                     data_layout);

#ifdef ONEDAL_DATA_PARALLEL
template ONEDAL_EXPORT void homogen_table::init_impl(const detail::data_parallel_policy&,
                                                     int64_t,
                                                     int64_t,
                                                     const array<byte_t>&,
                                                     const data_type&,
                                                     data_layout);
#endif

} // namespace oneapi::dal
