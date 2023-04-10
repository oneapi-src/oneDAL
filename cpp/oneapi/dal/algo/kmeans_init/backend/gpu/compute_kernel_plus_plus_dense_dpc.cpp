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

#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/placement.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"

namespace oneapi::dal::kmeans_init::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template<typename Float = double>
std::int64_t fix_trials_count(std::int64_t trial_count, 
                              std::int64_t sample_count) {
    ONEDAL_ASSERT(-1 <= trial_count);
    const auto additional = std::log<Float>(sample_count);
    const auto proposed = 2 + std::int64_t{ additional };
    return (trial_count == -1) ? proposed : trial_count;
}

template<typename Type>
sycl::event add_number(sycl::queue& queue,
                       const Type& value,
                       ndview<Type, 1>& array,
                       const event_vector& deps = {}) {
    constexpr std::plus<Type> kernel{};
    return pr::element_wise(queue, kernel, array, value, array, deps);
}

template<typename Type>
sycl::event minimum(sycl::queue& queue,
                    const ndview<Type, 1>& inp1,
                    const ndview<Type, 1>& inp2,
                    ndview<Type, 1>& output,
                    const event_vector& deps = {}) {
    constexpr std::plus<Type> kernel{};
    ONEDAL_ASSERT(inp1.get_count() == inp2.get_count());
    ONEDAL_ASSERT(inp1.get_count() == output.get_count());
    return pr::element_wise(queue, kernel, inp1, inp2, output, deps);
}

template <typename Type>
std::int64_t find_bin(const dal::array<Type>& offsets, const Type& value) {
    ONEDAL_ASSERT(offsets.has_data());
    const auto* const last = bk::cend(offsets);
    const auto* const first = bk::cbegin(offsets);
    
    ONEDAL_ASSERT(std::is_sorted(first, last));
    const auto iter = std::lower_bound(first, last, value);
    return dal::integral_cast<std::int64_t>(std::distance(first, iter)); 
}

template <typename Comm, typename Type>
auto get_boundaries(Comm& comm, const Type& local) {
    const auto count = comm.get_rank_count();

    using res_t = dal::array<Type>;
    auto res = res_t::zeros(count + 1);

    auto* const last = bk::end(res);
    auto* const first = bk::begin(res);

    {
        auto view = res_t::wrap(first + 1, count);
        comm.allgather(local, view).wait();
    }

    std::partial_sum(first, last, first);
    ONEDAL_ASSERT(std::is_sorted(first, last));
    return res;
}

template <typename Float, ndorder order>
sycl::event extract_and_share_by_index(const bk::context_gpu& ctx,
                                       std::int64_t index,
                                       const dal::array<std::int64_t>& offsets, 
                                       const ndview<Float, 2, order>& input,
                                       ndview<Float, 1>& place, 
                                       const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(offsets.has_data());
    ONEDAL_ASSERT(place.has_mutable_data());

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    const auto rank = comm.get_rank();
    const auto target = find_bin(offsets, index);
    const auto sample_count = data.get_dimension(0);
    ONEDAL_ASSERT(data.get_dimension(1) == place.get_count());
    ONEDAL_ASSERT(offsets.get_count() == comm.get_rank_count() + 1);

    // Extract part
    auto new_deps{ deps };
    if (rank == target) {
        const auto start_index = offsets[rank];
        const auto last_index = offsets[rank + 1];
        const auto rel_index = index - start_index;

        ONEDAL_ASSERT(start_index <= index && index < last_index);
        ONEDAL_ASSERT(0 <= rel_index && rel_index < sample_count);
        
        const auto source = data.get_row_slice(rel_index, rel_index + 1);
        auto dest = place.template reshape<2>({ 1, place.get_count() });
        new_deps.push_back(pr::copy(queue, dest, source, deps));
    } 


    // Share part
    {
        auto dst = place.flatten(queue, new_deps);
        comm.bcast(dst, target).wait();
    }

    return bk::wait_or_pass(new_deps);
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
                              const pr::ndview<Float, 2>& distances,
                              const pr::ndview<std::int32_t, 2>& indices,
                              const bk::event_vector& deps) -> sycl::event {
        ONEDAL_ASSERT(indices.has_data() && distances.has_data());
        ONEDAL_ASSERT(distances.get_dimension(1) == std::int64_t{ 1 });

        const auto last = blocking.get_block_end_index(qb_id);
        const auto first = blocking.get_block_start_index(qb_id);
        const auto block_length = blocking.get_block_length(qb_id);
        ONEDAL_ASSERT(distances.get_dimension(0) == block_length);

        const auto output_slice = closest_distances.get_slice(first, last);
        auto distance_2d = distances.template reshape<1>({ block_length });

        return minimum(queue, output_slice, distance_2d, distance_2d, deps);
    };

    return search_object(samples, callback, query_block, std::int64_t{ 1 }, deps);
}

template <typename Generator, typename Float>
void generate_trials(Generator& rng, dal::array<Float>& res, Float pot) {
    conststd::uniform_real_distribution<Float> finalize(0, pot);
    const auto gen = [&]() -> Float { return finalize(rng); };
    std::generate(bk::begin(trls), bk::end(trls), gen);
}

template <typename Generator, typename Float, ndorder order>
sycl::event first_sample(const bk::context_gpu& ctx, 
                         Generator& rng,
                         std::int64_t full_count,
                         const pr::ndview<Float, 2, order>& data,
                         pr::ndview<Float, 2>& centroids,
                         const event_vector& deps = {}) {
    const auto sample_count = data.get_dimension(0);
    const auto feature_count = dsta.get_dimension(1);
    ONEDAL_ASSERT(feature_count == centroids.get_dimension(1));
    const std::uniform_int_distribution<std::int64_t> finalize(0, full_count);
    auto slice = centroids.row_slice(0, 1).template reshape<1>({ feature_count });
    return extract_and_share(ctx, finalize(rng), boundaries, data, slice, deps);
}

template <typename Float, ndorder order>
Float compute_potential(sycl::queue& queue,
                        std::int64_t curr_count,
                        const pr::ndview<Float, 2>& centroids,
                        const pr::ndview<Float, 2, order>& data,
                        pr::ndview<Float, 1>& distances,
                        const event_vector& deps = {}) {
    const auto feature_count = data.get_dimension(1);
    ONEDAL_ASSERT(curr_count <= centroids.get_dimension(0));
    ONEDAL_ASSERT(feature_count == centroids.get_dimension(1));

    auto dist_slice = distance.get_slice(0, curr_count);
    auto cent_slice = centroids.get_row_slice(0, curr_count)
                        .template reshape<1>({ feature_count });

    using dist_l2 = pr::squared_l2_distance<Float>;
    using search_l2 =  pr::search_engine<Float, dist_l2, order>;
    auto train_block = pr::propose_train_block(queue, feature_count);

    const search_l2 search_object{ queue, cent_slice, train_block };
    auto closest_event = find_local_closest(queue, search_object, data, dist_slice, deps);

    auto minimum_event = minimum();
}

template <typename Method, typename Task, typename Float, pr::ndorder order>
compute_result<Task> implementation(const bk::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const pr::ndview<Float, 2, order>& data,
                                    const event_vector& deps = {}) {
    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    auto rng = std::mt19937_64(params.get_seed());
    const auto cluster_count = params.get_cluster_count();
    const auto trials_count = fix_trials_count(params.get_local_trials_count());

    const auto sample_count = data.get_dimension(0);
    const auto feature_count = data.get_dimension(1);

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count <= sample_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    auto centroids_array = dal::array<Float>::empty(queue, 
        dal::detail::check_mul_overflow(cluster_count, feature_count), alloc);
    auto centroids = pr::ndview<Float, 2>::wrap(centroids_array.get_mutable_data(), 
                                                { cluster_count, feature_count });

    auto boundaries = get_boundaries(comm, sample_count);
    const auto full_count = boundaries[comm.get_rank_count()];

    auto last_event = first_sample(ctx, rng, full_count, data, centroids, deps);


    auto trials = dal::array<Float>::empty(trials_count);
    auto host_trials = pr::ndview<Float, 1>::wrap(trials);

    auto distances = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);

    auto potential = compute_potential() 
    for (std::int64_t i = 1; i < cluster_count; ++i) {
        generate_trials(rng, trials, potential);

        compute
        auto device_trials = host_trials.to_device(queue, { last_event });
        auto search_event = pr::search_sorted(queue, )
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
