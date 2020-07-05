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
#include "oneapi/dal/data/table.hpp"

using std::int64_t;

namespace oneapi::dal {

template <typename Data>
homogen_table::homogen_table(sycl::queue& queue,
                             int64_t row_count,
                             int64_t column_count,
                             const Data* data_pointer,
                             homogen_data_layout layout,
                             const sycl::vector_class<sycl::event>& dependencies) {
    detail::wait_and_throw(dependencies);
    init_impl(backend::homogen_table_impl(row_count, column_count, data_pointer, layout));
}

template ONEAPI_DAL_EXPORT homogen_table::homogen_table(sycl::queue&,
                                                        int64_t,
                                                        int64_t,
                                                        const float*,
                                                        homogen_data_layout,
                                                        const sycl::vector_class<sycl::event>&);
template ONEAPI_DAL_EXPORT homogen_table::homogen_table(sycl::queue&,
                                                        int64_t,
                                                        int64_t,
                                                        const double*,
                                                        homogen_data_layout,
                                                        const sycl::vector_class<sycl::event>&);
template ONEAPI_DAL_EXPORT homogen_table::homogen_table(sycl::queue&,
                                                        int64_t,
                                                        int64_t,
                                                        const std::int32_t*,
                                                        homogen_data_layout,
                                                        const sycl::vector_class<sycl::event>&);

} // namespace oneapi::dal
