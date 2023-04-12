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
    ONEDAL_ASSERT(array.has_mutable_data());
    return element_wise(queue, kernel, array, value, array, deps);
}

template<typename Type, typename ax1, typename ax2>
sycl::event min_number(sycl::queue& queue,
                       ndview<Type, ax1>& array,
                       const ndview<Type, ax2> minimum,
                       const event_vector& deps = {}) {
    constexpr std::min<Type> kernel{};
    ONEDAL_ASSERT(array.has_mutable_data());
    return element_wise(queue, kernel, array, minimum, array, deps);
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

        auto output_slice = closest_distances.get_slice(first, last);
        auto distance_1d = distances.template reshape<1>({ block_length });
        return minimum(queue, output_slice, distance_1d, output_slice, deps);
    };

    return search_object(samples, callback, query_block, std::int64_t{ 1 }, deps);
}

template <typename Generator, typename Float>
void generate_trials(Generator& rng, pr::ndview<Float, 1>& res, Float pot) {
    const std::uniform_real_distribution<Float> finalize(0, pot);
    const auto gen = [&]() -> Float { return finalize(rng); };
    std::generate(bk::begin(trls), bk::end(trls), gen);
}

template <typename Generator, typename Float, ndorder order>
sycl::event first_sample(const bk::context_gpu& ctx, 
                         Generator& rng,
                         std::int64_t full_count,
                         pr::ndview<Float, 2>& centroids,
                         const pr::ndview<Float, 2, order>& samples,
                         pr::ndview<Float, 1>& distances,
                         pr::ndview<Float, 1>& candidates_norm,
                         const pr::ndview<Float, 1>& samples_norm,
                         const event_vector& deps = {}) {
    constexpr pr::sum<Float> sum{};
    constexpr pr::square<Float> square{};

    auto& queue = ctx.get_queue();
    const auto sample_count = samples.get_dimension(0);
    const auto feature_count = samples.get_dimension(1);
    ONEDAL_ASSERT(feature_count == centroids.get_dimension(1));

    ONEDAL_ASSERT(centroids.has_mutable_data());
    auto slice_2d = centroids.row_slice(0, 1);
    auto slice = slice_2d.template reshape<1>({ feature_count });
    const std::uniform_int_distribution<std::int64_t> finalize(0, full_count);
    auto share_event = extract_and_share(ctx, finalize(rng), boundaries, samples, slice, deps);
    
    ONEDAL_ASSERT(distances.has_mutable_data());
    ONEDAL_ASSERT(candidates_norm.has_mutable_data());
    auto first_norm = candidates_norm.get_slice(0, 1);
    auto fill_event = pr::fill(queue, first_norm, Float(0), deps);
    auto dist_2d = distances.template reshape<2>({ 1, sample_count });
    auto norm_event = pr::reduce_by_rows(queue, slice_2d, first_norm, sum, square, { fill_event });
    return compute_distances_to_cluster(queue, slice_2d, samples, dist_2d, first_norm, samples_norm, { share_event, norm_event });
}

template <typename Float>
std::int64_t propose_sample_block(const sycl::queue& queue, std::int64_t feature_count) {
    return 4096l;
}

template <typename Float>
std::int64_t propose_candidate_block(const sycl::queue& queue, std::int64_t feature_count) {
    return 4096l;
}

template <typename Float, pr::ndorder order> 
sycl::event compute_potential_local(sycl::queue& queue,
                                    const pr::ndview<Float, 1>& closest,
                                    const pr::ndview<Float, 2>& candidates,
                                    const pr::ndview<Float, 2, order>& samples,
                                    pr::ndview<Float, 2>& distances,
                                    pr::ndview<Float, 1>& potential,
                                    const pr::ndview<Float, 1>& candidate_norms,
                                    const pr::ndview<Float, 1>& sample_norms,
                                    const event_vector& deps = {}) {
    constexpr pr::sum<Float> sum{};
    constexpr pr::identity<Float> identity{};
    
    ONEDAL_ASSERT(candidates.has_data() && samples.has_data());
    ONEDAL_ASSERT(candidate_norms.has_data() && sample_norms.has_data());
    ONEDAL_ASSERT(distances.has_mutable_data() && potential.has_mutable_data());

    const auto sample_count = samples.get_dimension(0);
    ONEDAL_ASSERT(sample_count == closest.get_count());
    ONEDAL_ASSERT(sample_count == sample_norms.get_count());
    const auto feature_count = candidates.get_dimension(1);
    ONEDAL_ASSERT(feature_count == samples.get_dimension(1));
    const auto candidate_count = candidates.get_dimension(1);
    ONEDAL_ASSERT(candidate_count == potential.get_count());
    ONEDAL_ASSERT(candidate_count == candidate_norms.get_count());

    const auto sample_block = distances.get_dimension(1);
    const auto candidate_block = distances.get_dimension(0);

    const pr::distance<Float, pr::squared_l2_metric<Float>> dist_l2{ queue };

    const bk::uniform_blocking sample_blocking(sample_count, sample_block);
    const bk::uniform_blocking candidate_blocking(candidate_count, candidate_block);

    sycl::event last_event = pr::fill(queue, potential, Float(0), deps);

    for (std::int64_t cd_id = 0; cd_id < candidate_blocking.get_block_count(); ++cb_id) {
        const auto cd_first = candidate_blocking.get_block_start_index(cd_id);
        const auto cd_last = candidate_blocking.get_block_end_index(cd_id);

        const auto candidate_slice = candidates.get_row_slice(cd_first, cd_last);
        const auto candidate_norms_slice = candidate_norms.get_slice(cd_first, cd_last);

        auto dist_cd_slice = distances.get_row_slice(cd_first, cd_last);
        auto pot_cd_slice = potential.get_slice(cd_first, cd_last);

        for (std::int64_t sp_id = 0; sp_id < sample_blocking.get_block_count(); ++sp_id) {
            const auto sp_first = sample_blocking.get_block_start_index(sp_id);
            const auto sp_last = sample_blocking.get_block_end_index(sp_id);

            const auto closest_slice = closest.get_slice(sp_first, sp_last);
            const auto sample_slice = samples.get_row_slice(sp_first, sp_last);
            const auto sample_norms_slice = samples_norms.get_slice(sp_first, sp_last);

            auto dist_sp_slice = dist_cd_slice.get_col_slice(sp_first, sp_last);

            auto dist_event = dist_l2(candidate_slice, sample_slice, dist_sp_slice, 
                        candidate_norms_slice, sample_norms_slice, { last_event });


            auto min_event = min_number(queue,
                                        dist_sp_slice, 
                                        closest_slice,
                                        { dist_event });

            last_event = pr::reduce_by_rows(queue, 
                                            dist_sp_slice, 
                                            pot_cd_slice,
                                            sum,
                                            identity,
                                            { min_event });
        }
    }

    return last_event;
}

template <typename Float, pr::ndorder order> 
sycl::event compute_potential(const bk::context_gpu& ctx,
                              const pr::ndview<Float, 1>& closest,
                              const pr::ndview<Float, 2>& candidates,
                              const pr::ndview<Float, 2, order>& samples,
                              pr::ndview<Float, 2>& distances,
                              pr::ndview<Float, 1>& potential,
                              const pr::ndview<Float, 1>& candidate_norms,
                              const pr::ndview<Float, 1>& sample_norms,
                              const event_vector& deps = {}) {
    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    auto local_event = compute_potential_local(queue,
                                               closest,
                                               candidates,
                                               samples,
                                               distances,
                                               potential,
                                               candidate_norms,
                                               sample_norms,
                                               deps);

    if (comm.get_rank_count() > 0) {
        sycl::event::wait_and_throw({ local_event });
        auto potential_arr = dal::array<Float>::wrap(queue, 
            potential.get_mutable_data(), potential.get_count());
        comm.allreduce(potential_arr).wait();
    }

    return local_event;
}

template <typename Float, pr::ndorder order>
sycl::event compute_distances_to_cluster(sycl::queue& queue,
                                         const ndview<Float, 1>& cluster,
                                         const ndview<Float, 2, order>& samples,
                                         ndview<Float, 1>& output,
                                         const ndview<Float, 1>& cluster_norm,
                                         const ndview<Float, 1>& samples_norm,
                                         const event_vector& deps = {}) {
    ONEDAL_ASSERT(output.has_mutable_data());
    ONEDAL_ASSERT(cluster.has_data() && samples.has_data());
    ONEDAL_ASSERT(cluster_norm.has_data() && samples_norm.has_data());

    const auto sample_count = samples.get_dimension(0);
    const auto feature_count = samples.get_dimension(1);
    ONEDAL_ASSERT(sample_count == output.get_count());
    ONEDAL_ASSERT(feature_count == cluster.get_count());
    ONEDAL_ASSERT(sample_count == samples_norm.get_count());
    ONEDAL_ASSERT(std::int64_t(1) == cluster_norm.get_count());

    const pr::distance<Float, pr::squared_l2_metric<Float>> dist_l2{ queue };
    const auto cluster_2d = cluster.template reshape<2>({ 1, feature_count });
    auto output_2d = output.template reshape<2>({ 1, sample_count });

    return dist_l2(cluster_2d, samples, output_2d, cluster_norm, samples_norm, deps);
}

template <typename Method, typename Task, typename Float, pr::ndorder order>
compute_result<Task> implementation(const bk::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const pr::ndview<Float, 2, order>& samples,
                                    const event_vector& deps = {}) {
    constexpr pr::square<Float> square{};
    constexpr pr::identity<Float> identity{};

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    auto rng = std::mt19937_64(params.get_seed());
    const auto cluster_count = params.get_cluster_count();
    const auto trials_count = fix_trials_count(params.get_local_trials_count());

    const auto sample_count = samples.get_dimension(0);
    const auto feature_count = samples.get_dimension(1);

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count <= sample_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr auto alignment = search_alignment::left;
    constexpr auto max_val = std::numeric_limits<Float>::max();
    auto res_count = dal::detail::check_mul_overflow(cluster_count, feature_count);
    auto centroids_array = dal::array<Float>::full(queue, res_count, max_val, alloc);
    auto centroids = pr::ndview<Float, 2>::wrap(centroids_array.get_mutable_data(), 
                                                { cluster_count, feature_count });

    auto boundaries = get_boundaries(comm, sample_count);
    const auto full_count = boundaries[comm.get_rank_count()];

    auto trials = dal::array<Float>::empty(trials_count);
    auto host_trials = pr::ndview<Float, 1>::wrap(trials);

    auto last_event = first_sample(ctx, rng, full_count, samples, centroids, deps);
    auto distances = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);

    const auto sample_block = propose_sample_block(queue, feature_count);
    const auto candidate_block = propose_candidate_block(queue, feature_count);
    auto distances = pr::ndarray<Float, 2>::empty(queue, { candidate_block, sample_block }, alloc);

    auto candidates = pr::ndarray<Float, 2>::empty(queue, {trials_count, feature_count}, alloc);
    auto candidate_norms = pr::ndarray<Float, 1>::empty(queue, { trials_count }, alloc);

    auto sample_norms = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);
    auto sample_norms_fill = pr::fill(queue, sample_norms, Float(0));
    auto sample_norms_event = pr::reduce_by_rows(queue, 
                                                 samples,
                                                 sample_norms,
                                                 { sample_norms_fill });

                    

    auto potential = compute_potential() 
    for (std::int64_t i = 1; i < cluster_count; ++i) {
        generate_trials(rng, host_trials, potential);

        auto cumsum_event = pr::cumulative_sum_1d(queue, distances, {});
        const auto local = distances.at_device(queue, sample_count - 1, { cumsum_event });

        const auto boundaries = get_boundaries(comm, local);
        const auto local_offset = boundaries[ comm.get_rank() ]; 
        auto adjust_event = add_number(queue, local_offset, distances);

        auto device_trials = host_trials.to_device(queue, { last_event });
        auto search_event = pr::search_sorted(queue, )

        auto potentials = 


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
