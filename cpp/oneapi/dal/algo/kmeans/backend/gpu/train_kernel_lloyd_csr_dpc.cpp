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

#include <tuple>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/cluster_updater.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/kmeans/detail/train_init_centroids.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;
using event_vector = std::vector<sycl::event>;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;
namespace bk = dal::backend;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_init_plus_plus_csr_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<daal_kmeans_init::plusPlusCSR, Float, Cpu>;

// Initializes centroids randomly on CPU if it was not set by user.
template <typename Float, typename Method>
static pr::ndarray<Float, 2> get_initial_centroids(const dal::backend::context_gpu& ctx,
                                                   const descriptor_t& params,
                                                   const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();

    const auto data = static_cast<const csr_table&>(input.get_data());

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    if (!input.get_initial_centroids().has_data()) {
        auto daal_initial_centroids = daal_generate_centroids<Float, Method>(params, data);
        daal::data_management::BlockDescriptor<Float> block;
        daal_initial_centroids->getBlockOfRows(0,
                                               cluster_count,
                                               daal::data_management::readOnly,
                                               block);
        Float* initial_centroids_ptr = block.getBlockPtr();
        auto arr_host_initial =
            pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
        return arr_host_initial.to_device(queue);
    }
    auto initial_centroids_ptr = row_accessor<const Float>(input.get_initial_centroids())
                                     .pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    return pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
}

template <typename Float>
sycl::event compute_data_squares(sycl::queue& q,
                                 const dal::array<Float>& values,
                                 const dal::array<std::int64_t>& column_indices,
                                 const dal::array<std::int64_t>& row_offsets,
                                 pr::ndarray<Float, 1>& squares) {
    const auto local_size = bk::device_max_wg_size(q);
    const auto row_count = row_offsets.get_count() - 1;
    const std::int64_t row_block = 8 * bk::device_max_wg_size(q);
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ row_block, local_size }, { 1, local_size });
    const auto data_ptr = values.get_data();
    const auto row_offsets_ptr = row_offsets.get_data();
    auto squares_ptr = squares.get_mutable_data();
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_shift = item.get_global_id(0);
            const auto local_id = item.get_local_id(1);
            Float sum_squares = Float(0);
            for (std::int64_t row_idx = row_shift; row_idx < row_count; row_idx += row_block) {
                const auto row_start = row_offsets_ptr[row_idx] + local_id;
                const auto row_end = row_offsets_ptr[row_idx + 1];
                for (std::int64_t data_idx = row_start; data_idx < row_end;
                     data_idx += local_size) {
                    const auto val = data_ptr[data_idx];
                    sum_squares += val * val;
                }
                const auto res = sycl::reduce_over_group(item.get_group(),
                                                         sum_squares,
                                                         Float(0),
                                                         sycl::ext::oneapi::plus<Float>());
                if (local_id == 0) {
                    squares_ptr[row_idx] = res;
                }
            }
        });
    });
}

// Temporary function, TODO: replace this call with spgemm call
// TODO: need to add dimensions integer overflow
template <typename Float>
sycl::event custom_spgemm(sycl::queue& q,
                          const dal::array<Float>& values,
                          const dal::array<std::int64_t>& column_indices,
                          const dal::array<std::int64_t>& row_offsets,
                          const pr::ndarray<Float, 2>& b,
                          pr::ndarray<Float, 2>& c,
                          const Float alpha,
                          const Float beta,
                          const event_vector& deps = {}) {
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

/// Calculates distances to centroid and select closest centroid to each point
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] values           A data part of csr table
/// @param[in] column_indices   An array of column indices in csr table
/// @param[in] row_offsets      An arrat of row offsets in csr table
/// @param[in] data_squares     An array of data squared elementwise
/// @param[out] distances       An array of distances of dataset to each cluster
/// @param[out] centroids       An array of centroids with :expr:`cluster_count x column_count` dimensions
/// @param[out] closest_dists   An array of closests distances for each data point
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
sycl::event assign_clusters(sycl::queue& q,
                            const dal::array<Float>& values,
                            const dal::array<std::int64_t>& column_indices,
                            const dal::array<std::int64_t>& row_offsets,
                            const pr::ndarray<Float, 1>& data_squares,
                            const pr::ndarray<Float, 2>& centroids,
                            const pr::ndarray<Float, 1>& centroid_squares,
                            pr::ndarray<Float, 2>& distances,
                            pr::ndarray<std::int32_t, 2>& responses,
                            pr::ndarray<Float, 2>& closest_dists,
                            const event_vector& deps = {}) {
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
    const std::int64_t row_block = 8 * bk::device_max_wg_size(q);

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

// Counts the number of points for each cluster.
// Result is cluster_counts array and returning value is the number of empty clusters.
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] responses        An array of cluster assignments
/// @param[out] cluster_count   As a result of function call the number of data points for each cluster
/// @param[in] deps             An event vector of dependencies for specified kernel
std::int32_t count_clusters(sycl::queue& q,
                            const pr::ndarray<std::int32_t, 2>& responses,
                            pr::ndarray<std::int32_t, 1>& cluster_counts,
                            const event_vector& deps = {}) {
    auto device_cluster_count = pr::ndarray<std::int32_t, 1>::empty(q, 1, sycl::usm::alloc::device);
    auto device_count_ptr = device_cluster_count.get_mutable_data();

    const auto row_count = responses.get_dimension(0);
    const auto resp_ptr = responses.get_data();
    const auto num_cluster = cluster_counts.get_dimension(0);
    auto cluster_count_glob = cluster_counts.get_mutable_data();
    const auto local_size = bk::device_max_wg_size(q);
    const auto range = bk::make_multiple_nd_range_1d(local_size, local_size);
    auto event = q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<std::int32_t, 1> local_clust_counts(num_cluster, cgh);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](auto it) {
            std::int32_t* const counts =
                local_clust_counts.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            const auto local_id = it.get_local_id(0);
            if (static_cast<std::int64_t>(local_id) < num_cluster) {
                counts[local_id] = 0;
            }
            it.barrier();
            for (std::int32_t row_idx = local_id; row_idx < row_count; row_idx += local_size) {
                const auto cluster_id = resp_ptr[row_idx];
                sycl::atomic_ref<std::int32_t,
                                 sycl::memory_order_relaxed,
                                 sycl::memory_scope_work_group,
                                 sycl::access::address_space::local_space>
                    cl_count(local_clust_counts[cluster_id]);
                cl_count += 1;
            }
            it.barrier();
            if (static_cast<std::int64_t>(local_id) < num_cluster) {
                cluster_count_glob[local_id] = counts[local_id];
            }
            if (local_id == 0) {
                std::int32_t empty_count = 0;
                for (std::int32_t cl_id = 0; cl_id < num_cluster; ++cl_id) {
                    if (counts[cl_id] == 0) {
                        empty_count++;
                    }
                }
                device_count_ptr[0] = empty_count;
            }
        });
    });
    auto host_empty_cluster_count = device_cluster_count.to_host(q, { event }).get_data();
    return host_empty_cluster_count[0];
}

// Calculates an objective function, which is sum of all distances from points to centroid.
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] dists            An array of distances for each data point to the closest cluster
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
Float calc_objective_function(sycl::queue& q,
                              const pr::ndview<Float, 2>& dists,
                              const event_vector& deps = {}) {
    pr::sum<Float> sum{};
    pr::identity<Float> ident{};
    auto view_1d = dists.template reshape<1>(pr::ndshape<1>{dists.get_dimension(0)});
    return pr::reduce_1d(q, view_1d, sum, ident, deps);
}

// Updates centroids based on new responses and cluster counts.
// New centroid is a mean among all points in cluster.
// If cluster is empty, centroid remains the same as in previous iteration.
/// @param[in] q                A sycl-queue to perform operations on device
/// @param[in] values           A data part of csr table
/// @param[in] column_indices   An array of column indices in csr table
/// @param[in] row_offsets      An arrat of row offsets in csr table
/// @param[in] column_count     A number of column in input dataset
/// @param[in] reponses         An array of cluster assignments
/// @param[out] centroids       An array of centroids with :expr:`cluster_count x column_count` dimensions
/// @param[in] cluster_counts   An array of cluster counts
/// @param[in] deps             An event vector of dependencies for specified kernel
template <typename Float>
sycl::event update_centroids(sycl::queue& q,
                             const bk::communicator<spmd::device_memory_access::usm>& comm,
                             const dal::array<Float>& values,
                             const dal::array<std::int64_t>& column_indices,
                             const dal::array<std::int64_t>& row_offsets,
                             std::int64_t column_count,
                             const pr::ndarray<std::int32_t, 2>& responses,
                             pr::ndarray<Float, 2>& centroids,
                             const pr::ndarray<std::int32_t, 1>& cluster_counts,
                             const event_vector& deps = {}) {
    const auto resp_ptr = responses.get_data();
    auto centroids_ptr = centroids.get_mutable_data();
    const auto row_count = row_offsets.get_count() - 1;
    const auto data_ptr = values.get_data();
    const auto row_ofs_ptr = row_offsets.get_data();
    const auto col_ind_ptr = column_indices.get_data();
    const auto counts_ptr = cluster_counts.get_data();

    const auto local_size = bk::device_max_wg_size(q);
    const auto num_clusters = centroids.get_dimension(0);
    const auto range = bk::make_multiple_nd_range_3d({ num_clusters, column_count, local_size },
                                                     { 1, 1, local_size });
    auto centroids_sum_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](auto it) {
            const auto cluster_id = it.get_global_id(0);
            const auto col_idx = it.get_global_id(1);
            const auto local_id = it.get_local_id(2);

            Float acc = 0;
            for (std::int32_t row_idx = local_id; row_idx < row_count; row_idx += local_size) {
                if (resp_ptr[row_idx] == static_cast<std::int32_t>(cluster_id)) {
                    const auto start = row_ofs_ptr[row_idx];
                    const auto end = row_ofs_ptr[row_idx + 1];
                    for (std::int32_t data_idx = start; data_idx < end; data_idx++) {
                        if (col_ind_ptr[data_idx] == static_cast<std::int64_t>(col_idx)) {
                            acc += data_ptr[data_idx];
                        }
                    }
                }
            }
            const auto final_component =
                sycl::reduce_over_group(it.get_group(), acc, sycl::ext::oneapi::plus<Float>());
            // if count for cluster is zero, the centroid will remain as previous one
            if (local_id == 0 && counts_ptr[cluster_id] > 0) {
                centroids_ptr[cluster_id * column_count + col_idx] = final_component;
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
/// @param[out] responses       An array of cluster assignments
/// @param[out] cluster_counts  An array of cluster counts
/// @param[out] dists           An array of closest distances to cluster
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


/// Main entrypoint for GPU CSR Kmeans algorithm
/// @param[in] ctx          GPU context structure
/// @param[in] params       A descriptor containing parameters for algorithm
/// @param[in] input        A train input
template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_csr, task::clustering> {
    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const train_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        auto& comm = ctx.get_communicator();
        ONEDAL_ASSERT(input.get_data().get_kind() == dal::csr_table::kind());
        const auto data = static_cast<const csr_table&>(input.get_data());
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t cluster_count = params.get_cluster_count();
        const std::int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();
        dal::detail::check_mul_overflow(cluster_count, column_count);

        auto [values, column_indices, row_offsets] =
            csr_accessor<const Float>(data).pull(queue,
                                                 { 0, -1 },
                                                 sparse_indexing::zero_based,
                                                 sycl::usm::alloc::device);

        auto arr_initial = get_initial_centroids<Float, method::lloyd_csr>(ctx, params, input);
        auto arr_centroid_squares =
            pr::ndarray<Float, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);
        auto arr_data_squares =
            pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto data_squares_event =
            compute_data_squares(queue, values, column_indices, row_offsets, arr_data_squares);

        auto distances = pr::ndarray<Float, 2>::empty(queue,
                                                      { row_count, cluster_count },
                                                      sycl::usm::alloc::device);

        auto arr_closest_distances =
            pr::ndarray<Float, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                          { cluster_count, column_count },
                                                          sycl::usm::alloc::device);
        auto arr_responses =
            pr::ndarray<std::int32_t, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto cluster_counts =
            pr::ndarray<std::int32_t, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);

        Float prev_objective_function = de::limits<Float>::max();
        std::int64_t iter;
        sycl::event last_event = data_squares_event;

        for (iter = 0; iter < max_iteration_count; iter++) {
            auto centroid_squares_event =
                kernels_fp<Float>::compute_squares(queue,
                                                   iter == 0 ? arr_initial : arr_centroids,
                                                   arr_centroid_squares,
                                                   { last_event });
            auto assign_event = assign_clusters(queue,
                                                values,
                                                column_indices,
                                                row_offsets,
                                                arr_data_squares,
                                                iter == 0 ? arr_initial : arr_centroids,
                                                arr_centroid_squares,
                                                distances,
                                                arr_responses,
                                                arr_closest_distances,
                                                { centroid_squares_event, last_event });
            count_clusters(queue, arr_responses, cluster_counts, { assign_event });

            {
                // Cluster counters over all ranks in case of distributed computing
                auto count_reduce_event =
                    comm.allreduce(cluster_counts.flatten(queue, { assign_event }));
                count_reduce_event.wait();
            }

            auto empty_cluster_event = handle_empty_clusters(ctx,
                                                             row_count,
                                                             arr_responses,
                                                             cluster_counts,
                                                             arr_closest_distances,
                                                             { assign_event });

            auto objective_function =
                calc_objective_function(queue,
                                        arr_closest_distances,
                                        { empty_cluster_event, assign_event });

            {
                // Reduce objective function value over all ranks
                auto obj_func_reduce_event = comm.allreduce(objective_function);
                obj_func_reduce_event.wait();
            }
            auto update_event = update_centroids(queue,
                                                 comm,
                                                 values,
                                                 column_indices,
                                                 row_offsets,
                                                 column_count,
                                                 arr_responses,
                                                 arr_centroids,
                                                 cluster_counts,
                                                 { assign_event });

            last_event = update_event;

            if (accuracy_threshold > 0 &&
                objective_function + accuracy_threshold > prev_objective_function) {
                iter++;
                break;
            }
            prev_objective_function = objective_function;
        }
        auto centroid_squares_event =
            kernels_fp<Float>::compute_squares(queue,
                                               iter == 0 ? arr_initial : arr_centroids,
                                               arr_centroid_squares,
                                               { last_event });
        auto assign_event = assign_clusters(queue,
                                            values,
                                            column_indices,
                                            row_offsets,
                                            arr_data_squares,
                                            iter == 0 ? arr_initial : arr_centroids,
                                            arr_centroid_squares,
                                            distances,
                                            arr_responses,
                                            arr_closest_distances,
                                            { last_event, centroid_squares_event });
        auto objective_function =
            calc_objective_function(queue,
                                    arr_closest_distances,
                                    { last_event, centroid_squares_event, assign_event });
        {
            // Reduce objective function value over all ranks
            auto obj_func_reduce_event = comm.allreduce(objective_function);
            obj_func_reduce_event.wait();
        }

        model<task::clustering> model;
        model.set_centroids(
            dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count));
        return train_result<task::clustering>()
            .set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1))
            .set_iteration_count(iter)
            .set_objective_function_value(objective_function)
            .set_model(model);
    }
};

template struct train_kernel_gpu<float, method::lloyd_csr, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_csr, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
