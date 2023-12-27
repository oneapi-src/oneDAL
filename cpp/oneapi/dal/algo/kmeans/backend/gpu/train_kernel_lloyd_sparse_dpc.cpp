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

#include <tuple>
#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/cluster_updater.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
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

template <typename Float>
static pr::ndarray<Float, 2> get_initial_centroids(const dal::backend::context_gpu& ctx,
                                                   const descriptor_t& params,
                                                   const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();

    const auto data = static_cast<const csr_table&>(input.get_data());

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    daal::data_management::NumericTablePtr daal_initial_centroids;

    if (!input.get_initial_centroids().has_data()) {
        // We use CPU algorithm for initialization, so input data
        // may be copied to DAAL homogen table
        const auto daal_data = interop::copy_to_daal_csr_table<Float>(data);
        daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count));

        const std::size_t init_len_input = 1;
        daal::data_management::NumericTable* init_input[init_len_input] = { daal_data.get() };

        daal_initial_centroids =
            interop::allocate_daal_homogen_table<Float>(cluster_count, column_count);
        const std::size_t init_len_output = 1;
        daal::data_management::NumericTable* init_output[init_len_output] = {
            daal_initial_centroids.get()
        };

        const dal::backend::context_cpu cpu_ctx;
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_kmeans_init_plus_plus_csr_kernel_t>(
                cpu_ctx,
                init_len_input,
                init_input,
                init_len_output,
                init_output,
                &par,
                *(par.engine)));
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
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ row_count, local_size }, { 1, local_size });
    const auto data_ptr = values.get_data();
    const auto row_offsets_ptr = row_offsets.get_data();
    auto squares_ptr = squares.get_mutable_data();
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_idx = item.get_global_id(0);
            const auto local_id = item.get_local_id(1);
            Float sum_squares = Float(0);
            const auto row_start = row_offsets_ptr[row_idx] + local_id;
            const auto row_end = row_offsets_ptr[row_idx + 1];
            for (std::int64_t data_idx = row_start; data_idx < row_end; data_idx += local_size) {
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
    const auto a_row_count = row_offsets.get_count() - 1;
    const auto reduce_dim = b.get_dimension(1);
    const auto b_row_count = b.get_dimension(0);
    ONEDAL_ASSERT(c.get_dimension(0) == a_row_count);
    ONEDAL_ASSERT(c.get_dimension(1) == b_row_count);

    const auto local_size =
        std::min<std::int32_t>(bk::device_max_wg_size(q), bk::down_pow2(reduce_dim));
    auto res_ptr = c.get_mutable_data();
    const auto a_ptr = values.get_data();
    const auto row_ofs = row_offsets.get_data();
    const auto col_ind = column_indices.get_data();
    const auto b_ptr = b.get_data();

    const auto nd_range = bk::make_multiple_nd_range_3d({ a_row_count, b_row_count, local_size },
                                                        { 1, 1, local_size });

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_idx = item.get_global_id(0);
            const auto col_idx = item.get_global_id(1);
            const auto local_id = item.get_local_id(2);

            const auto start = row_ofs[row_idx] + local_id;
            const auto end = row_ofs[row_idx + 1];
            Float acc = Float(0);
            for (std::int64_t data_idx = start; data_idx < end; data_idx += local_size) {
                const auto reduce_id = col_ind[data_idx];
                acc += a_ptr[data_idx] * b_ptr[col_idx * reduce_dim + reduce_id];
            }
            const Float scalar_mul = sycl::reduce_over_group(item.get_group(),
                                                             acc,
                                                             Float(0),
                                                             sycl::ext::oneapi::plus<Float>());
            if (local_id == 0) {
                res_ptr[row_idx * b_row_count + col_idx] =
                    beta * res_ptr[row_idx * b_row_count + col_idx] + alpha * scalar_mul;
            }
        });
    });
}

// Calculates distances to centroid and select closest centroid to each point
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
    const auto row_count = row_offsets.get_count() - 1;

    const auto local_size =
        std::min<std::int64_t>(bk::device_max_wg_size(q), bk::down_pow2(cluster_count));
    const auto nd_range =
        bk::make_multiple_nd_range_2d({ row_count, local_size }, { 1, local_size });

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ dist_event });
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](auto item) {
            const auto row_idx = item.get_global_id(0);
            const auto local_id = item.get_local_id(1);
            const auto max_val = std::numeric_limits<Float>::max();
            const auto max_index = std::numeric_limits<std::int32_t>::max();
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
        });
    });
    return event;
}

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

template <typename Float>
Float calc_objective_function(sycl::queue& q,
                              const pr::ndarray<Float, 2>& dists,
                              const event_vector& deps = {}) {
    auto res = pr::ndarray<Float, 1>::empty(q, 1, sycl::usm::alloc::device);
    auto res_ptr = res.get_mutable_data();
    const auto row_count = dists.get_dimension(0);
    const auto dists_ptr = dists.get_data();
    const auto local_size =
        std::min<std::int32_t>(bk::device_max_wg_size(q), bk::down_pow2(row_count));
    const auto range = bk::make_multiple_nd_range_1d(local_size, local_size);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](auto it) {
            const auto local_id = it.get_local_id(0);
            Float sum = 0;
            for (std::int32_t row_idx = local_id; row_idx < row_count; row_idx += local_size) {
                sum += dists_ptr[row_idx];
            }
            const Float obj_func =
                sycl::reduce_over_group(it.get_group(), sum, sycl::ext::oneapi::plus<Float>());
            if (local_id == 0) {
                res_ptr[0] = obj_func;
            }
        });
    });
    auto host_res = res.to_host(q, { event });
    return host_res.get_data()[0];
}

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
        // Cluster counters over all ranks in case of distributed computing
        auto count_reduce_event = comm.allreduce(cluster_counts.flatten(q, { deps }));
        count_reduce_event.wait();
    }
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

template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_sparse, task::clustering> {
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

        auto arr_initial = get_initial_centroids<Float>(ctx, params, input);
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
            auto objective_function =
                calc_objective_function(queue, arr_closest_distances, { assign_event });

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

template struct train_kernel_gpu<float, method::lloyd_sparse, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_sparse, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
