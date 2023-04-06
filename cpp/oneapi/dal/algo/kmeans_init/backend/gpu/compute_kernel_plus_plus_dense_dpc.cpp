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

#include <numeric>

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/placement.hpp"

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"

namespace oneapi::dal::kmeans_init::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, ndorder order>
sycl::event extract_and_share_by_index(const bk::context_gpu& ctx,
                                       std::int64_t index,
                                       std::int64_t start_index,
                                       const ndview<Float, 2, order>& data,
                                       ndview<Float, 1>& place, 
                                       const event_vector& deps = {}) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(place.has_mutable_data());

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    const auto sample_count = data.get_dimension(0);
    ONEDAL_ASSERT(data.get_dimension(1) == place.get_count());

}

template <typename Comm, typename Type>
Type get_local_offset(Comm& comm,
                      const Type& value,
                      const dal::array<Type>& temp) {
    ONEDAL_ASSERT(temp.has_mutable_data());
    const auto rank_count = comm.get_rank_count();
    ONEDAL_ASSERT(rank_count == temp.get_count());

    if (rank_count > 1) {
        comm.allgather(value, temp).wait();
    } 
    else {
        return value;
    }

    const auto* const ptr = temp.get_data();
    return std::reduce(ptr, ptr + comm.get_rank());
}

template <typename Comm, typename Type>
Type get_local_offset(Comm& comm,
                      const Type& value) {
    const auto rank_count = comm.get_rank_count();
    if (rank_count == 1) return value;
    auto temp = dal::array<Type>::empty(rank_count);
    return get_local_offset(comm, value, temp);
}

template <typename SearchObject, typename Float>
sycl::event find_local_closest(sycl::queue& queue,
                               const SearchObject& search_object,
                               const pr::ndview<Float, 2>& samples,
                               pr::ndview<Float, 1>& closest_distances,
                               const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(samples.has_data());
    ONEDAL_ASSERT(closest_distances.has_mutable_data());
    const auto feature_count = samples.get_dimension(1);
    const auto sample_count = samples.get_dimension(0);

    ONEDAL_ASSERT(sample_count == closest_distances.get_count());
    auto query_block = pr::propose_query_block(queue, sample_count);

    const bk::uniform_blocking blocking(sample_count, query_block);
    const auto callback = [&](std::int64_t qb_id,
                              const auto& indices,
                              const auto& distances,
                              const bk::event_vector& dependencies) {
        ONEDAL_ASSERT(indices.has_data() && distances.has_data());
        ONEDAL_ASSERT(distances.get_dimension(1) == std::int64_t{ 1 });

        const auto last = blocking.get_block_end_index(qb_id);
        const auto first = blocking.get_block_start_index(qb_id);
        const auto block_length = blocking.get_block_length(qb_id);
        ONEDAL_ASSERT(distances.get_dimension(0) == block_length);

        const auto output_slice = closest_distances.get_slice(first, last);
        auto output_2d = output_slice.template reshape<1>({ block_length, 1 });

        return pr::copy(queue, output_2d, distances, dependencies);
    };

    return search_object(samples, callback, query_block, 1, deps);
}

template <typename Method, typename Task, typename Float, pr::ndorder order>
inline compute_result<Task> implementation(const bk::context_gpu& ctx,
                                           const detail::descriptor_base<Task>& params,
                                           const pr::ndview<Float, 2, order>& data) {
    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    const auto seed = params.get_seed();
    const auto cluster_count = params.get_cluster_count();

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count <= sample_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    auto centroids_array = array<Float>::empty(queue, 
        dal::detail::check_mul_overflow(cluster_count, feature_count), alloc);
    auto centroids = pr::ndview<Float, 2>::wrap(centroids_array.get_mutable_data(), 
                                                { cluster_count, feature_count });


    for (std::int64_t i = 1; i < cluster_count; ++i) {
    }

    auto centroids_array = 
    return compute_result<Task>{}.set_centroids(homogen_table::wrap(centroids, cluster_count, feature_count));
}

template <typename Float, typename Method, typename Task>
compute_result<Task> compute_kernel_distr<Float, Method, Task>::operator()(
    const bk::context_gpu& ctx,
    const detail::descriptor_base<Task>& params,
    const compute_input<Task>& input) const {

    auto& queue = ctx.get_queue();
    const auto& data_table = input.get_data();
    constexpr auto alloc = sycl::usm::alloc::device;

    auto data_variant = pr::table2ndarray_variant<Float>(queue, data_table, alloc);

    return std::visit([&](const auto& data) -> compute_result<Task> {
        return implementation<Method, Task>(ctx, params, data);
    }, data_variant);
}

template struct compute_kernel_distr<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_distr<double, method::plus_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
