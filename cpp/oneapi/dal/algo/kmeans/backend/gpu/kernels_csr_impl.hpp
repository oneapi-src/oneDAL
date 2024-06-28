/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/backend/atomic.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;
using event_vector = std::vector<sycl::event>;

template <typename Data>
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;
namespace bk = dal::backend;

template <typename Float>
sycl::event compute_data_squares(sycl::queue& q,
                                 const pr::ndview<Float, 1>& values,
                                 const pr::ndview<std::int64_t, 1>& column_indices,
                                 const pr::ndview<std::int64_t, 1>& row_offsets,
                                 pr::ndview<Float, 1>& squares,
                                 const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_data_squares, q);
    return pr::reduce_by_rows(q,
                              values,
                              column_indices,
                              row_offsets,
                              sparse_indexing::zero_based,
                              squares,
                              pr::sum<Float>{},
                              pr::square<Float>{},
                              deps);
}

template <typename Float>
sycl::event transpose(sycl::queue& q,
                      const pr::ndview<Float, 2>& src,
                      pr::ndview<Float, 2>& dst,
                      const event_vector& deps = {}) {
    const auto src_shape = src.get_shape();
    const auto row_count = src_shape[0];
    const auto col_count = src_shape[1];

    const auto nd_range = sycl::range<2>(row_count, col_count);

    const Float* src_ptr = src.get_data();
    Float* dst_ptr = dst.get_mutable_data();
    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            auto i = item[0];
            auto j = item[1];
            dst_ptr[j * row_count + i] = src_ptr[i * col_count + j];
        });
    });

    return event;
}

template <typename Float>
sycl::event assign_clusters(sycl::queue& q,
                            const std::size_t row_count,
                            pr::sparse_matrix_handle& data_handle,
                            const pr::ndview<Float, 1>& data_squares,
                            const pr::ndview<Float, 2>& centroids_transposed,
                            const pr::ndview<Float, 1>& centroid_squares,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<std::int32_t, 2>& responses,
                            pr::ndview<Float, 2>& closest_dists,
                            const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(assign_clusters, q);
    // Calculate rest part of distances
    auto dist_event = pr::gemm(q,
                               pr::transpose::nontrans,
                               data_handle,
                               centroids_transposed,
                               distances,
                               Float(-2.0),
                               Float(0),
                               deps);

    auto selection_event = kernels_fp<Float>::select(q,
                                                     distances,
                                                     centroid_squares,
                                                     closest_dists,
                                                     responses,
                                                     { dist_event });
    return kernels_fp<Float>::complete_closest_distances(q, data_squares, closest_dists, { selection_event });
}

// Calculates an objective function, which is sum of all distances from points to centroid.
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] dists            An array of distances for each data point to the closest cluster
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
Float calc_objective_function(sycl::queue& q,
                              const pr::ndview<Float, 2>& dists,
                              const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(calc_objective_function, q);
    pr::sum<Float> sum{};
    pr::identity<Float> ident{};
    auto view_1d = dists.template reshape<1>(pr::ndshape<1>{ dists.get_dimension(0) });
    return pr::reduce_1d(q, view_1d, sum, ident, deps);
}

// Updates the centroids based on new responses and cluster counts.
// New centroid is a mean among all points in cluster.
// If cluster is empty, centroid remains the same as in previous iteration.
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] values           A data part of csr table with :expr:`non_zero_count x 1` dimensions
/// @param[in] column_indices   An array of column indices in csr table :expr:`non_zero_count x 1` dimensions
/// @param[in] row_offsets      An arrat of row offsets in csr table with :expr:`(row_count + 1) x 1` dimensions
/// @param[in] column_count     A number of column in input dataset
/// @param[in] responses        An array of cluster assignments with :expr:`row_count x 1` dimensions
/// @param[out] centroids       An array of centroids with :expr:`cluster_count x column_count` dimensions
/// @param[in] cluster_counts   An array of cluster counts with :expr:`cluster_count x 1` dimensions
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
sycl::event update_centroids(sycl::queue& q,
                             const pr::ndview<Float, 1>& values,
                             const pr::ndview<std::int64_t, 1>& column_indices,
                             const pr::ndview<std::int64_t, 1>& row_offsets,
                             const std::int64_t column_count,
                             const pr::ndarray<std::int32_t, 2>& responses,
                             pr::ndarray<Float, 2>& centroids,
                             const pr::ndarray<std::int32_t, 1>& cluster_counts,
                             const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_centroids, q);
    const auto resp_ptr = responses.get_data();
    auto centroids_ptr = centroids.get_mutable_data();
    const auto row_count = row_offsets.get_count() - 1;
    const auto data_ptr = values.get_data();
    const auto row_ofs_ptr = row_offsets.get_data();
    const auto col_ind_ptr = column_indices.get_data();
    const auto counts_ptr = cluster_counts.get_data();

    const auto num_clusters = centroids.get_dimension(0);

    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, num_clusters, column_count);
    const auto centroids_elem_count = num_clusters * column_count;
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, centroids_elem_count, sizeof(Float));
    const auto centroids_num_bytes = centroids_elem_count * sizeof(Float);

    auto clean_event = q.memset(centroids_ptr, 0, centroids_num_bytes);

    const auto wg_size = 16;
    const size_t row_count_unsigned = static_cast<size_t>(row_count);

    const size_t wg_count = (row_count + wg_size - 1) / wg_size;

    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, wg_size, wg_count);
    const std::int64_t global_size = wg_count * wg_size;
    auto range = bk::make_multiple_nd_range_2d( { global_size, num_clusters }, { wg_size, 1 } );

    auto centroids_sum_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(clean_event);
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_centroids(wg_size * column_count, cgh);
        cgh.parallel_for(range, [=](sycl::nd_item<2> it) {
            const auto global_row_id = it.get_global_id(0);
            const auto global_centorid_id = it.get_global_id(1);
            if (global_row_id >= row_count_unsigned)
                return;

            const size_t centorid_id = resp_ptr[global_row_id];

            const auto local_id = it.get_local_id(0);

            Float * local_accessor_ptr =
                local_centroids.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float * local_centroid_ptr = local_accessor_ptr + local_id * column_count;

            for (auto idx = 0; idx < column_count; ++idx) {
                local_centroid_ptr[idx] = 0;
            }

            if (global_centorid_id == centorid_id) {
                const auto begin_idx = row_ofs_ptr[global_row_id];
                const auto end_idx = row_ofs_ptr[global_row_id + 1];
                for (auto idx = begin_idx; idx < end_idx; ++idx) {
                    const auto col_idx = col_ind_ptr[idx];
                    local_centroid_ptr[col_idx] = data_ptr[idx];
                }
            }

            // reduction for loop
            for (size_t i = wg_size / 2; i > 0; i >>= 1) {
                it.barrier(sycl::access::fence_space::local_space);
                if (local_id < i) {
                    for (auto idx = 0; idx < column_count; ++idx) {
                        local_centroid_ptr[idx] += local_centroid_ptr[i * column_count + idx];
                    }
                }
            }

            if (local_id == 0) {
                for (auto idx = 0; idx < column_count; ++idx) {
                    bk::atomic_global_add(centroids_ptr + global_centorid_id * column_count + idx,
                                          local_centroid_ptr[idx]);
                }
            }
        });
    });

    const auto finalize_range =
        bk::make_multiple_nd_range_2d({ num_clusters, wg_size }, { 1, wg_size });
    auto finalize_centroids = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(centroids_sum_event);
        cgh.parallel_for(finalize_range, [=](auto it) {
            const auto cluster_id = it.get_global_id(0);
            const auto local_id = it.get_local_id(1);
            const auto cent_count = counts_ptr[cluster_id];
            if (cent_count == 0)
                return;

            for (std::int32_t col_idx = local_id; col_idx < column_count; col_idx += wg_size) {
                centroids_ptr[cluster_id * column_count + col_idx] /= cent_count;
            }
        });
    });

    return finalize_centroids;
}

} // namespace oneapi::dal::kmeans::backend
