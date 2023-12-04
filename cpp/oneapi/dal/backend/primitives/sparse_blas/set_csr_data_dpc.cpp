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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/set_csr_data.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle_iface.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event set_csr_data(sycl::queue& queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         dal::sparse_indexing indexing,
                         dal::array<Float> &data,
                         dal::array<std::int64_t> &column_indices,
                         dal::array<std::int64_t> &row_offsets,
                         const std::vector<sycl::event> &deps)
{
    return oneapi::mkl::sparse::set_csr_data(
        queue,
        dal::detail::get_impl(handle).handle,
        row_count,
        column_count,
        sparse_indexing_to_mkl(indexing),
        const_cast<std::int64_t *>(row_offsets.get_data()),
        const_cast<std::int64_t *>(column_indices.get_data()),
        const_cast<Float *>(data.get_data()),
        deps);
}

template <typename Float>
sycl::event set_csr_data(sycl::queue& queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         dal::sparse_indexing indexing,
                         const Float * data,
                         const std::int64_t * column_indices,
                         const std::int64_t * row_offsets,
                         const std::vector<sycl::event> &deps)
{
    return oneapi::mkl::sparse::set_csr_data(
        queue,
        dal::detail::get_impl(handle).handle,
        row_count,
        column_count,
        sparse_indexing_to_mkl(indexing),
        const_cast<std::int64_t *>(row_offsets),
        const_cast<std::int64_t *>(column_indices),
        const_cast<Float *>(data),
        deps);
}

template <typename Float>
sycl::event set_csr_data(sycl::queue& queue,
                         sparse_matrix_handle &handle,
                         dal::csr_table &table,
                         const sycl::usm::alloc& alloc,
                         const std::vector<sycl::event> &deps) {
    const std::int64_t row_count = table.get_row_count();
    const std::int64_t column_count = table.get_column_count();
    const dal::sparse_indexing indexing = table.get_indexing();

    const auto [data_array, cidx_array, ridx_array] = csr_accessor<const Float>(table).pull(
                queue,
                { 0, row_count },
                indexing,
                alloc);

    return set_csr_data(queue,
                        handle,
                        row_count,
                        column_count,
                        indexing,
                        data_array.get_data(),
                        cidx_array.get_data(),
                        ridx_array.get_data(),
                        deps);
}

#define INSTANTIATE(F)                                                                              \
    template ONEDAL_EXPORT sycl::event set_csr_data<F>(sycl::queue& queue,                          \
                                                       sparse_matrix_handle &handle,                \
                                                       const std::int64_t row_count,                \
                                                       const std::int64_t column_count,             \
                                                       dal::sparse_indexing indexing,               \
                                                       dal::array<F> &data,                         \
                                                       dal::array<std::int64_t> &column_indices,    \
                                                       dal::array<std::int64_t> &row_offsets,       \
                                                       const std::vector<sycl::event> &deps);       \
                                                                                                    \
    template ONEDAL_EXPORT sycl::event set_csr_data<F>(sycl::queue& queue,                          \
                                                       sparse_matrix_handle &handle,                \
                                                       const std::int64_t row_count,                \
                                                       const std::int64_t column_count,             \
                                                       dal::sparse_indexing indexing,               \
                                                       const F * data,                              \
                                                       const std::int64_t * column_indices,         \
                                                       const std::int64_t * row_offsets,            \
                                                       const std::vector<sycl::event> &deps);       \
                                                                                                    \
    template ONEDAL_EXPORT sycl::event set_csr_data<F>(sycl::queue& queue,                          \
                                                       sparse_matrix_handle &handle,                \
                                                       dal::csr_table &table,                       \
                                                       const sycl::usm::alloc& alloc,               \
                                                       const std::vector<sycl::event> &deps);


INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
