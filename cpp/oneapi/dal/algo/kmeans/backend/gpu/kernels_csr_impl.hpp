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

// Temporary function, TODO: replace this call with spgemm call
// TODO: need to add dimensions integer overflow
template <typename Float>
sycl::event custom_spgemm(sycl::queue& q,
                          const pr::ndview<Float, 1>& values,
                          const pr::ndview<std::int64_t, 1>& column_indices,
                          const pr::ndview<std::int64_t, 1>& row_offsets,
                          const pr::ndview<Float, 2>& b,
                          pr::ndview<Float, 2>& c,
                          const Float alpha,
                          const Float beta,
                          const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(custom_spgemm, q);
    const size_t a_row_count = row_offsets.get_count() - 1;
    const size_t reduce_dim = b.get_dimension(1);
    const size_t b_row_count = b.get_dimension(0);

    const auto local_size =
        std::min<std::int32_t>(bk::device_max_wg_size(q), bk::down_pow2(reduce_dim));
    auto res_ptr = c.get_mutable_data();
    const auto a_ptr = values.get_data();
    const auto row_ofs = row_offsets.get_data();
    const auto col_ind = column_indices.get_data();
    const auto b_ptr = b.get_data();

    // Compute matrix block by block to avoid integer overflow
    const std::int64_t row_block = 8 * bk::device_max_wg_size(q);
    const std::int64_t row_block_size = std::min<std::int64_t>(row_block, a_row_count);
    const std::int64_t col_block_size = std::min<std::int64_t>(row_block, b_row_count);

    const auto nd_range =
        bk::make_multiple_nd_range_3d({ row_block_size, col_block_size, local_size },
                                      { 1, 1, local_size });

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_shift = item.get_global_id(0);
            const auto col_shift = item.get_global_id(1);
            const auto local_id = item.get_local_id(2);

            for (auto row_idx = row_shift; row_idx < a_row_count; row_idx += row_block) {
                for (auto col_idx = col_shift; col_idx < b_row_count; col_idx += row_block) {
                    const auto start = row_ofs[row_idx] + local_id;
                    const auto end = row_ofs[row_idx + 1];
                    Float acc = Float(0);
                    for (std::int64_t data_idx = start; data_idx < end; data_idx += local_size) {
                        const auto reduce_id = col_ind[data_idx];
                        acc += a_ptr[data_idx] * b_ptr[col_idx * reduce_dim + reduce_id];
                    }
                    const Float scalar_mul =
                        sycl::reduce_over_group(item.get_group(),
                                                acc,
                                                Float(0),
                                                sycl::ext::oneapi::plus<Float>());
                    if (local_id == 0) {
                        res_ptr[row_idx * b_row_count + col_idx] =
                            beta * res_ptr[row_idx * b_row_count + col_idx] + alpha * scalar_mul;
                    }
                }
            }
        });
    });
}

/// Calculates distances from each data point to each centroid and selects the closest centroid to each data point
/// @param[in] q                    A sycl-queue to perform operations on device
/// @param[in] values               A data part of csr table with :expr:`non_zero_count x 1` dimensions
/// @param[in] column_indices       An array of column indices in csr table with :expr:`non_zero_count x 1` dimensions
/// @param[in] row_offsets          An arrat of row offsets in csr table with :expr:`(row_count + 1) x 1` dimensions
/// @param[in] data_squares         An array of data squared elementwise with :expr:`row_count x 1` dimensions
/// @param[in] centroids           An array of centroids with :expr:`cluster_count x column_count` dimensions
/// @param[in] centroids_squares   An array of centroids squares with :expr:`cluster_count x 1` dimensions
/// @param[out] distances           An array of distances of dataset to each cluster with :expr:`row_count x cluster_count` dimensions
/// @param[out] responses           An array of responses with :expr:`row_count x 1` dimensions
/// @param[out] closest_dists       An array of closests distances for each data point with :expr:`row_count x 1` dimensions
/// @param[in] deps                 An event vector of dependencies for specified kernel
template <typename Float>
sycl::event assign_clusters(sycl::queue& q,
                            const pr::ndview<Float, 1>& values,
                            const pr::ndview<std::int64_t, 1>& column_indices,
                            const pr::ndview<std::int64_t, 1>& row_offsets,
                            const pr::ndview<Float, 1>& data_squares,
                            const pr::ndview<Float, 2>& centroids,
                            const pr::ndview<Float, 1>& centroid_squares,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<std::int32_t, 2>& responses,
                            pr::ndview<Float, 2>& closest_dists,
                            const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(assign_clusters, q);
    auto data_squares_ptr = data_squares.get_data();
    auto cent_squares_ptr = centroid_squares.get_data();
    auto responses_ptr = responses.get_mutable_data();
    auto closest_dists_ptr = closest_dists.get_mutable_data();
    // Calculate rest part of distances
    auto dist_event = custom_spgemm(q,
                                    values,
                                    column_indices,
                                    row_offsets,
                                    centroids,
                                    distances,
                                    Float(-2.0),
                                    Float(0),
                                    deps);

    const auto distances_ptr = distances.get_data();

    const auto cluster_count = centroids.get_dimension(0);
    const auto row_count = static_cast<size_t>(row_offsets.get_count() - 1);
    // based on bechmarks an optimal block size is equal to 8 work-group sizes
    const std::int64_t block_multiplier = 8;
    const std::int64_t row_block = block_multiplier * bk::device_max_wg_size(q);

    const auto local_size =
        std::min<std::int64_t>(bk::device_max_wg_size(q), bk::down_pow2(cluster_count));
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ row_block, local_size }, { 1, local_size });

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ dist_event });
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_shift = item.get_global_id(0);
            const auto local_id = item.get_local_id(1);
            const auto max_val = std::numeric_limits<Float>::max();
            const auto max_index = std::numeric_limits<std::int32_t>::max();
            for (auto row_idx = row_shift; row_idx < row_count; row_idx += row_block) {
                auto min_dist = max_val;
                auto min_idx = max_index;
                auto row_dists = distances_ptr + row_idx * cluster_count;
                for (std::int32_t cluster_id = local_id; cluster_id < cluster_count;
                     cluster_id += local_size) {
                    const auto dist = cent_squares_ptr[cluster_id] + row_dists[cluster_id] +
                                      data_squares_ptr[row_idx];
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = cluster_id;
                    }
                }
                const Float closest = sycl::reduce_over_group(item.get_group(),
                                                              min_dist,
                                                              max_val,
                                                              sycl::ext::oneapi::minimum<Float>());
                const std::int32_t dist_idx = closest == min_dist ? min_idx : max_index;
                const std::int32_t closest_id =
                    sycl::reduce_over_group(item.get_group(),
                                            dist_idx,
                                            max_index,
                                            sycl::ext::oneapi::minimum<std::int32_t>());
                if (local_id == 0) {
                    responses_ptr[row_idx] = closest_id;
                    closest_dists_ptr[row_idx] = closest;
                }
            }
        });
    });
    return event;
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
    auto data_squares_ptr = data_squares.get_data();
    auto cent_squares_ptr = centroid_squares.get_data();
    auto responses_ptr = responses.get_mutable_data();
    auto closest_dists_ptr = closest_dists.get_mutable_data();
    // Calculate rest part of distances
    auto dist_event = pr::gemm(q,
                               pr::transpose::nontrans,
                               data_handle,
                               centroids_transposed,
                               distances,
                               Float(-2.0),
                               Float(0),
                               deps);

    const auto distances_ptr = distances.get_data();

    const auto cluster_count = centroids_transposed.get_dimension(1);
    // based on bechmarks an optimal block size is equal to 8 work-group sizes
    const std::int64_t block_multiplier = 8;
    const std::int64_t row_block = block_multiplier * bk::device_max_wg_size(q);

    const auto local_size =
        std::min<std::int64_t>(bk::device_max_wg_size(q), bk::down_pow2(cluster_count));
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ row_block, local_size }, { 1, local_size });

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ dist_event });
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_shift = item.get_global_id(0);
            const auto local_id = item.get_local_id(1);
            const auto max_val = std::numeric_limits<Float>::max();
            const auto max_index = std::numeric_limits<std::int32_t>::max();
            for (auto row_idx = row_shift; row_idx < row_count; row_idx += row_block) {
                auto min_dist = max_val;
                auto min_idx = max_index;
                auto row_dists = distances_ptr + row_idx * cluster_count;
                for (std::int32_t cluster_id = local_id; cluster_id < cluster_count;
                     cluster_id += local_size) {
                    const auto dist = cent_squares_ptr[cluster_id] + row_dists[cluster_id] +
                                      data_squares_ptr[row_idx];
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = cluster_id;
                    }
                }
                const Float closest = sycl::reduce_over_group(item.get_group(),
                                                              min_dist,
                                                              max_val,
                                                              sycl::ext::oneapi::minimum<Float>());
                const std::int32_t dist_idx = closest == min_dist ? min_idx : max_index;
                const std::int32_t closest_id =
                    sycl::reduce_over_group(item.get_group(),
                                            dist_idx,
                                            max_index,
                                            sycl::ext::oneapi::minimum<std::int32_t>());
                if (local_id == 0) {
                    responses_ptr[row_idx] = closest_id;
                    closest_dists_ptr[row_idx] = closest;
                }
            }
        });
    });
    return event;
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
/// @param[in] reponses         An array of cluster assignments with :expr:`row_count x 1` dimensions
/// @param[out] centroids       An array of centroids with :expr:`cluster_count x column_count` dimensions
/// @param[in] cluster_counts   An array of cluster counts with :expr:`cluster_count x 1` dimensions
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
sycl::event update_centroids(sycl::queue& q,
                             const bk::communicator<spmd::device_memory_access::usm>& comm,
                             const pr::ndview<Float, 1>& values,
                             const pr::ndview<std::int64_t, 1>& column_indices,
                             const pr::ndview<std::int64_t, 1>& row_offsets,
                             std::int64_t column_count,
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

    const auto local_size = bk::device_max_wg_size(q);
    const auto num_clusters = centroids.get_dimension(0);

    const auto clean_range =
        bk::make_multiple_nd_range_2d({ num_clusters, column_count }, { 1, 1 });
    auto clean_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(clean_range, [=](auto it) {
            const auto cluster_id = it.get_global_id(0);
            const auto col_id = it.get_global_id(1);
            centroids_ptr[cluster_id * column_count + col_id] = 0;
        });
    });

    const auto row_block =
        std::min<std::int32_t>(bk::device_max_wg_size(q) * 8, bk::down_pow2(row_count));
    const auto col_block =
        std::min<std::int32_t>(bk::device_max_wg_size(q), bk::down_pow2(column_count));
    const auto range =
        bk::make_multiple_nd_range_3d({ num_clusters, row_block, col_block }, { 1, 1, col_block });

    auto centroids_sum_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(clean_event);
        local_accessor_rw_t<Float> local_centroid(column_count, cgh);
        cgh.parallel_for(range, [=](auto it) {
            const auto cluster_id = it.get_global_id(0);
            const auto row_shift = it.get_global_id(1);
            const auto local_id = static_cast<std::int64_t>(it.get_local_id(2));
            if (counts_ptr[cluster_id] == 0) {
                return;
            }
            auto local_centroid_ptr =
                local_centroid.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            for (std::int64_t col_idx = local_id; col_idx < column_count; col_idx += col_block) {
                local_centroid_ptr[col_idx] = 0;
            }
            it.barrier();
            for (std::int64_t row_idx = row_shift; row_idx < row_count; row_idx += row_block) {
                if (resp_ptr[row_idx] == static_cast<std::int32_t>(cluster_id)) {
                    const auto start = row_ofs_ptr[row_idx];
                    const auto end = row_ofs_ptr[row_idx + 1];
                    for (auto idx = start + local_id; idx < end; idx += col_block) {
                        const auto col_idx = col_ind_ptr[idx];
                        const auto val = data_ptr[idx];
                        bk::atomic_local_add(local_centroid_ptr + col_idx, val);
                    }
                }
            }
            it.barrier();
            if (local_id == 0) {
                for (std::int64_t col_idx = 0; col_idx < column_count; ++col_idx) {
                    const auto pos = cluster_id * column_count + col_idx;
                    bk::atomic_global_add(centroids_ptr + pos, local_centroid_ptr[col_idx]);
                }
            }
        });
    });
    {
        // Reduce centroids over all ranks in of distributed computing
        auto centroids_reduce_event = comm.allreduce(centroids.flatten(q, { centroids_sum_event }));
        centroids_reduce_event.wait();
    }

    const auto finalize_range =
        bk::make_multiple_nd_range_2d({ num_clusters, local_size }, { 1, local_size });
    auto finalize_centroids = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(centroids_sum_event);
        cgh.parallel_for(finalize_range, [=](auto it) {
            const auto cluster_id = it.get_global_id(0);
            const auto local_id = it.get_local_id(1);
            const auto cent_count = counts_ptr[cluster_id];
            if (cent_count == 0) {
                return;
            }
            for (std::int32_t col_idx = local_id; col_idx < column_count; col_idx += local_size) {
                centroids_ptr[cluster_id * column_count + col_idx] /= cent_count;
            }
        });
    });
    return finalize_centroids;
}

/// Handling empty clusters.
/// @param[in] ctx              GPU context structure
/// @param[in] row_count        A number of rows in the dataset
/// @param[out] responses       An array of cluster assignments with :expr:`row_count x 1` dimensions
/// @param[out] cluster_counts  An array of cluster counts with :expr:`cluster_count x 1` dimensions
/// @param[out] dists           An array of closest distances to cluster with :expr:`row_count x 1` dimensions
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
sycl::event handle_empty_clusters(const dal::backend::context_gpu& ctx,
                                  const std::int64_t row_count,
                                  pr::ndarray<std::int32_t, 2>& responses,
                                  pr::ndarray<std::int32_t, 1>& cluster_counts,
                                  pr::ndarray<Float, 2>& dists,
                                  const event_vector& deps = {}) {
    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();
    ONEDAL_PROFILER_TASK(handle_empty_clusters, queue);
    const auto rank_count = comm.get_rank_count();
    const auto rank = comm.get_rank();
    const auto num_clusters = cluster_counts.get_dimension(0);

    auto resp_ptr = responses.get_mutable_data();
    auto counts_ptr = cluster_counts.get_mutable_data();
    auto dists_ptr = dists.get_mutable_data();

    const auto abs_min_val = -std::numeric_limits<Float>::max();

    auto local_size = bk::device_max_wg_size(queue);
    auto range = bk::make_multiple_nd_range_1d(local_size, local_size);
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](auto it) {
            const auto local_id = it.get_local_id(1);
            for (std::int64_t cluster_id = rank; cluster_id < num_clusters;
                 cluster_id += rank_count) {
                // no need to handle non-empty clusters
                if (counts_ptr[cluster_id] > 0) {
                    continue;
                }
                std::int64_t cand_idx = -1;
                Float cand_dist = abs_min_val;
                for (std::int64_t row_idx = local_id; row_idx < row_count; row_idx += local_size) {
                    const auto dist = dists_ptr[row_idx];
                    if (dist > cand_dist) {
                        cand_dist = dist;
                        cand_idx = row_idx;
                    }
                }
                const Float longest_dist =
                    sycl::reduce_over_group(it.get_group(),
                                            cand_dist,
                                            abs_min_val,
                                            sycl::ext::oneapi::maximum<Float>());
                const auto id = longest_dist == cand_dist ? cand_idx : -1;
                const auto longest_id =
                    sycl::reduce_over_group(it.get_group(),
                                            id,
                                            sycl::ext::oneapi::maximum<std::int64_t>());
                if (local_id == 0 && longest_id != -1) {
                    resp_ptr[longest_id] = cluster_id;
                    counts_ptr[longest_id] = 1;
                    dists_ptr[cluster_id] = Float(0);
                }
            }
        });
    });
    return event;
}

} // namespace oneapi::dal::kmeans::backend
