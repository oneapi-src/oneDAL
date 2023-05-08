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
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/placement.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"

namespace oneapi::dal::kmeans_init::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

std::int64_t fix_trials_count(std::int64_t trial_count, std::int64_t cluster_count) {
    ONEDAL_ASSERT(trial_count > 0 || trial_count == -1);

    const auto additional = std::log(cluster_count);
    const auto proposed = 2 + std::int64_t(additional);
    auto result = (trial_count == -1) ? proposed : trial_count;

    ONEDAL_ASSERT(result > 0l);

    return result;
}

template <typename Type>
sycl::event add_number(sycl::queue& queue,
                       const pr::ndview<Type, 1>& value,
                       pr::ndview<Type, 1>& array,
                       const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(value.has_data());
    ONEDAL_ASSERT(array.has_mutable_data());

    Type* const arr_ptr = array.get_mutable_data();
    const auto range = bk::make_range_1d(array.get_count());

    const Type* const val_ptr = value.get_data();
    ONEDAL_ASSERT(value.get_count() == std::int64_t(1));

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<1> idx) {
            arr_ptr[idx] += *val_ptr;
        });
    });
}

template <typename Type, std::int64_t ax1, std::int64_t ax2>
sycl::event min_number(sycl::queue& queue,
                       pr::ndview<Type, ax1>& array,
                       const pr::ndview<Type, ax2> minimum,
                       const bk::event_vector& deps = {}) {
    constexpr sycl::minimum<Type> kernel{};
    ONEDAL_ASSERT(array.has_mutable_data());
    return element_wise(queue, kernel, array, minimum, array, deps);
}

template <typename Type>
std::int64_t find_bin(const dal::array<Type>& offsets, const Type& value) {
    const auto* const last = bk::cend(offsets);
    const auto* const first = bk::cbegin(offsets);

    ONEDAL_ASSERT(*first == Type(0));
    ONEDAL_ASSERT(std::is_sorted(first, last));

    const auto result = std::lower_bound(first, last, value);

    const auto curr = *result;
    const auto next =
        (result == std::prev(last)) ? std::numeric_limits<Type>::max() : *std::next(result);
    const auto prev = (result == first) ? std::numeric_limits<Type>::lowest() : *std::prev(result);

    ONEDAL_ASSERT((prev <= curr) && (curr <= next));

    if ((prev <= value) && (value < curr)) {
        return dal::detail::integral_cast<std::int64_t>( //
            std::distance(first, result) - 1);
    }

    if ((curr <= value) && (value < next)) {
        return dal::detail::integral_cast<std::int64_t>( //
            std::distance(first, result) + 0);
    }

    ONEDAL_ASSERT(false);

    return std::int64_t(-1);
}

template <typename Comm, typename Type>
dal::array<Type> get_boundaries(Comm& comm, const Type& local) {
    const auto count = comm.get_rank_count();
    auto res = dal::array<Type>::zeros(count + 1);

    auto* const last = bk::end(res);
    auto* const first = bk::begin(res);

    {
        auto view = dal::array<Type>::wrap(first + 1, count);
        comm.allgather(local, view).wait();
    }

    std::partial_sum(first, last, first);
    ONEDAL_ASSERT(std::is_sorted(first, last));
    return res;
}

template <typename Float, pr::ndorder order>
sycl::event extract_and_share_by_index(const bk::context_gpu& ctx,
                                       std::int64_t index,
                                       const dal::array<std::int64_t>& offsets,
                                       const pr::ndview<Float, 2, order>& data,
                                       pr::ndview<Float, 1>& place,
                                       const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(place.has_mutable_data());

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    const auto rank = comm.get_rank();
    const auto target = find_bin(offsets, index);
    const auto rank_count = comm.get_rank_count();
    ONEDAL_ASSERT(target <= rank_count);

    const auto sample_count = data.get_dimension(0);
    ONEDAL_ASSERT(data.get_dimension(1) == place.get_count());
    ONEDAL_ASSERT(offsets.get_count() == comm.get_rank_count() + 1);

    // Extract part
    bk::event_vector new_deps{ deps };
    if ((rank_count == 1l) || (rank == target)) {
        const auto start_index = offsets[rank];
        const auto last_index = offsets[rank + 1];
        const auto rel_index = index - start_index;

        ONEDAL_ASSERT(start_index <= index && index < last_index);
        ONEDAL_ASSERT(0 <= rel_index && rel_index < sample_count);

        const auto source = data.get_row_slice(rel_index, rel_index + 1);
        auto dest = place.template reshape<2>({ 1, source.get_count() });
        new_deps.push_back(pr::copy(queue, dest, source, deps));
    }

    // Share part
    if (rank_count > 1l) {
        const auto dst =
            array<Float>::wrap(queue, place.get_mutable_data(), place.get_count(), new_deps);
        comm.bcast(dst, target).wait();
    }

    return bk::wait_or_pass(new_deps);
}

template <typename Float, pr::ndorder order>
sycl::event extract_and_share_by_indices(const bk::context_gpu& ctx,
                                         const pr::ndview<std::int64_t, 1>& indices,
                                         const dal::array<std::int64_t>& offsets,
                                         const pr::ndview<Float, 2, order>& input,
                                         pr::ndview<Float, 2>& candidates,
                                         const bk::event_vector& deps = {}) {
    const auto candidate_count = candidates.get_dimension(0);
    const auto feature_count = candidates.get_dimension(1);
    ONEDAL_ASSERT(candidate_count == indices.get_count());

    ONEDAL_ASSERT(indices.has_data());
    auto slice_2d = candidates.get_row_slice(0, 1);
    auto slice = slice_2d.template reshape<1>({ feature_count });

    auto indices_host = indices.to_host(ctx.get_queue(), deps);
    auto last_event =
        extract_and_share_by_index(ctx, indices_host.at(0), offsets, input, slice, deps);

    for (std::int64_t i = 1; i < candidate_count; ++i) {
        auto result_2d = candidates.get_row_slice(i, i + 1);
        auto result = result_2d.template reshape<1>({ feature_count });
        last_event = extract_and_share_by_index(ctx,
                                                indices_host.at(i),
                                                offsets,
                                                input,
                                                result,
                                                { last_event });
    }

    return last_event;
}

template <typename Comm, typename Type>
Type get_local_offset(Comm& comm, const Type& value, const dal::array<Type>& temp) {
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
Type get_local_offset(Comm& comm, const Type& value) {
    const auto rank_count = comm.get_rank_count();
    if (rank_count == 1)
        return value;
    auto temp = dal::array<Type>::empty(rank_count);
    return get_local_offset(comm, value, temp);
}

template <typename Generator, typename Float, typename GenFloat = float>
void generate_trials(Generator& rng, pr::ndview<Float, 1>& trls, double pot) {
    std::uniform_real_distribution<GenFloat> finalize(0.0, pot);
    const auto gen = [&]() -> Float {
        return finalize(rng);
    };
    std::generate(bk::begin(trls), bk::end(trls), gen);
}

template <typename Float>
std::int64_t propose_sample_block(const sycl::queue& queue,
                                  std::int64_t feature_count,
                                  std::int64_t sample_count) {
    return std::min<std::int64_t>(sample_count, 4096l);
}

template <typename Float>
std::int64_t propose_candidate_block(const sycl::queue& queue,
                                     std::int64_t feature_count,
                                     std::int64_t trial_count) {
    return std::min<std::int64_t>(trial_count, 1024l);
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
                                    const bk::event_vector& deps = {}) {
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
    const auto candidate_count = candidates.get_dimension(0);
    ONEDAL_ASSERT(candidate_count == potential.get_count());
    ONEDAL_ASSERT(candidate_count == candidate_norms.get_count());

    const auto sample_block = distances.get_dimension(1);
    const auto candidate_block = distances.get_dimension(0);

    const pr::distance<Float, pr::squared_l2_metric<Float>> dist_l2{ queue };

    const bk::uniform_blocking sample_blocking(sample_count, sample_block);
    const bk::uniform_blocking candidate_blocking(candidate_count, candidate_block);

    sycl::event last_event = pr::fill(queue, potential, Float(0), deps);

    for (std::int64_t cd_id = 0; cd_id < candidate_blocking.get_block_count(); ++cd_id) {
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
            const auto sample_norms_slice = sample_norms.get_slice(sp_first, sp_last);

            auto dist_sp_slice = dist_cd_slice.get_col_slice(sp_first, sp_last);

            auto dist_event = dist_l2(candidate_slice,
                                      sample_slice,
                                      dist_sp_slice,
                                      candidate_norms_slice,
                                      sample_norms_slice,
                                      { last_event });

            auto min_event = min_number(queue, dist_sp_slice, closest_slice, { dist_event });

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
                              const bk::event_vector& deps = {}) {
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

    if (comm.get_rank_count() > 1) {
        sycl::event::wait_and_throw({ local_event });
        auto potential_arr =
            dal::array<Float>::wrap(queue, potential.get_mutable_data(), potential.get_count());
        comm.allreduce(potential_arr).wait();
    }

    return local_event;
}

template <typename Float, pr::ndorder order>
sycl::event compute_distances_to_cluster(sycl::queue& queue,
                                         const pr::ndview<Float, 1>& cluster,
                                         const pr::ndview<Float, 2, order>& samples,
                                         pr::ndview<Float, 1>& output,
                                         const pr::ndview<Float, 1>& cluster_norm,
                                         const pr::ndview<Float, 1>& samples_norm,
                                         const bk::event_vector& deps = {}) {
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

    auto dist_event = dist_l2(cluster_2d, samples, output_2d, cluster_norm, samples_norm, deps);

    return dist_event;
}

template <typename Generator, typename Float, pr::ndorder order>
sycl::event first_sample(const bk::context_gpu& ctx,
                         Generator& rng,
                         std::int64_t full_count,
                         pr::ndview<Float, 2>& centroids,
                         const dal::array<std::int64_t>& boundaries,
                         const pr::ndview<Float, 2, order>& samples,
                         pr::ndview<Float, 1>& potential,
                         pr::ndview<Float, 1>& distances,
                         pr::ndview<Float, 1>& candidates_norm,
                         const pr::ndview<Float, 1>& samples_norm,
                         const bk::event_vector& deps = {}) {
    constexpr pr::sum<Float> sum{};
    constexpr pr::square<Float> square{};
    constexpr pr::identity<Float> identity{};

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();
    const auto sample_count = samples.get_dimension(0);
    const auto feature_count = samples.get_dimension(1);
    ONEDAL_ASSERT(feature_count == centroids.get_dimension(1));

    ONEDAL_ASSERT(centroids.has_mutable_data());
    auto slice_2d = centroids.get_row_slice(0, 1);
    auto slice = slice_2d.template reshape<1>({ feature_count });
    std::uniform_int_distribution<std::int64_t> finalize(0, full_count - 1);

    const auto first_index = finalize(rng);
    auto share_event =
        extract_and_share_by_index(ctx, first_index, boundaries, samples, slice, deps);

    ONEDAL_ASSERT(potential.has_mutable_data());
    ONEDAL_ASSERT(distances.has_mutable_data());
    ONEDAL_ASSERT(candidates_norm.has_mutable_data());

    auto curr_potential = potential.get_slice(0, 1);
    auto first_norm = candidates_norm.get_slice(0, 1);
    auto pot_event = pr::fill(queue, potential, Float(0), deps);
    auto fill_event = pr::fill(queue, first_norm, Float(0), deps);
    auto dist_2d = distances.template reshape<2>({ 1, sample_count });
    auto norm_event =
        pr::reduce_by_rows(queue, slice_2d, first_norm, sum, square, { fill_event, share_event });
    auto dist_event = compute_distances_to_cluster(queue,
                                                   slice,
                                                   samples,
                                                   distances,
                                                   first_norm,
                                                   samples_norm,
                                                   { norm_event });
    auto local_event =
        pr::reduce_1d(queue, distances, curr_potential, sum, identity, { dist_event, pot_event });

    if (comm.get_rank_count() > 1) {
        sycl::event::wait_and_throw({ local_event });

        auto arr = array<Float>::wrap(queue, //
                                      curr_potential.get_mutable_data(),
                                      std::int64_t(1));
        comm.allreduce(arr).wait();
    }

    return local_event;
}

template <typename Float, typename Index>
sycl::event fix_indices(sycl::queue& queue,
                        const std::int64_t rank,
                        const std::int64_t offset,
                        const pr::ndview<Float, 1>& points,
                        const pr::ndview<Float, 1>& bounds,
                        pr::ndview<Index, 1>& indices,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(fix_indices.kernel, queue);

    constexpr auto max = std::numeric_limits<Float>::max();

    ONEDAL_ASSERT(points.has_data());
    ONEDAL_ASSERT(bounds.has_data());
    ONEDAL_ASSERT(indices.has_mutable_data());

    const auto count = points.get_count();
    const auto bnd_count = bounds.get_count();
    ONEDAL_ASSERT(count == indices.get_count());

    const auto range = bk::make_range_1d(count);
    const Float* const pts_ptr = points.get_data();
    const Float* const bds_ptr = bounds.get_data();
    Index* const ids_ptr = indices.get_mutable_data();

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<1> idx) {
            const auto value = pts_ptr[idx];
            const auto index = ids_ptr[idx];

            const auto curr_lower = bds_ptr[rank];
            const auto curr_upper = (bnd_count <= rank + 2) ? max : bds_ptr[rank + 1];

            const bool this_rank = (curr_lower <= value) && (value < curr_upper);

            ids_ptr[idx] = this_rank ? (index + offset) : std::numeric_limits<Index>::lowest();
        });
    });
}

template <typename Communicator, typename Float, typename Index>
sycl::event fix_indices(sycl::queue& queue,
                        Communicator& comm,
                        const std::int64_t offset,
                        const pr::ndview<Float, 1>& points,
                        const pr::ndview<Float, 1>& bounds,
                        pr::ndview<Index, 1>& indices,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(fix_indices, queue);

    const auto rank = comm.get_rank();
    const auto rank_count = comm.get_rank_count();
    ONEDAL_ASSERT(bounds.get_count() == rank_count + 1);
    auto event = fix_indices(queue, rank, offset, points, bounds, indices, deps);

    {
        ONEDAL_PROFILER_TASK(fix_indices, queue);
        sycl::event::wait_and_throw({ event });

        const auto ids_arr = array<std::int64_t>::wrap(queue, //
                                                       indices.get_mutable_data(),
                                                       indices.get_count(),
                                                       { event });

        comm.allreduce(ids_arr, dal::preview::spmd::reduce_op::max).wait();

        return sycl::event{};
    }
}

template <typename Communicator, typename Float, typename Index>
sycl::event find_indices(sycl::queue& queue,
                         Communicator& comm,
                         const std::int64_t offset,
                         const pr::ndview<Float, 1>& points,
                         pr::ndview<Float, 1>& values,
                         pr::ndview<Float, 1>& bounds,
                         pr::ndview<Index, 1>& indices,
                         const bk::event_vector& deps) {
    const auto rank = comm.get_rank();
    const auto rank_count = comm.get_rank_count();

    ONEDAL_ASSERT(bounds.has_mutable_data());
    ONEDAL_ASSERT((rank_count + 1) == bounds.get_count());

    ONEDAL_ASSERT(values.has_mutable_data());
    const auto local_count = values.get_count();

    // Computes local cumulative sum
    //auto fill_bnds_event = pr::fill(queue, bounds, Float(0), deps);
    auto last_event = pr::cumulative_sum_1d(queue, values, deps);
    auto fill_event = pr::fill(queue, bounds, Float(0), deps);

    const Float* const last_val_ptr = values.get_data() + local_count - 1;

    if (rank_count > 1) {
        ONEDAL_PROFILER_TASK(find_indices.boundaries, queue);

        auto bnds_arr = array<Float>::wrap(queue, bounds.get_mutable_data() + 1, rank_count);
        const Float* const last_val_ptr = values.get_data() + local_count - 1;
        auto last_val_arr = array<Float>::wrap(queue, last_val_ptr, 1);

        sycl::event::wait_and_throw({ last_event, fill_event });
        comm.allgather(last_val_arr, bnds_arr).wait();

        auto cumsum_event = pr::cumulative_sum_1d(queue, bounds);

        const auto adjustment = bounds.get_slice(rank, rank + 1);
        last_event = add_number(queue, adjustment, values, { cumsum_event });
    }
    else {
        auto slice = bounds.get_slice(rank + 1, rank + 2);
        const auto set = [=](auto&, auto* ptr) -> Float {
            return *ptr;
        };
        last_event =
            element_wise(queue, set, slice, last_val_ptr, slice, { last_event, fill_event });
    }

    constexpr auto alignment = pr::search_alignment::left;
    auto search_event = pr::search_sorted_1d(queue,
                                             alignment, //
                                             values,
                                             points,
                                             indices,
                                             { last_event });

    auto fix_event = fix_indices(queue, comm, offset, points, bounds, indices, { search_event });

    return fix_event;
}

template <typename Method, typename Task, typename Float, pr::ndorder order>
compute_result<Task> implementation(const bk::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const pr::ndview<Float, 2, order>& samples,
                                    const bk::event_vector& deps = {}) {
    using dal::backend::operator+;

    constexpr pr::sum<Float> sum{};
    constexpr pr::square<Float> square{};

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    auto rng = std::mt19937(params.get_seed());

    const auto sample_count = samples.get_dimension(0);
    const auto feature_count = samples.get_dimension(1);

    const auto cluster_count = params.get_cluster_count();
    const auto init_trial_count = params.get_local_trials_count();
    const auto trials_count = fix_trials_count(init_trial_count, cluster_count);

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count <= sample_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr auto max_val = std::numeric_limits<Float>::max();
    auto res_count = dal::detail::check_mul_overflow(cluster_count, feature_count);
    auto centroids_array = dal::array<Float>::full(queue, res_count, max_val, alloc);
    auto centroids = pr::ndview<Float, 2>::wrap(centroids_array.get_mutable_data(),
                                                { cluster_count, feature_count });

    const auto rank_count = comm.get_rank_count();

    auto boundaries = get_boundaries(comm, sample_count);
    const std::int64_t full_count = boundaries[rank_count];
    const std::int64_t curr_offset = boundaries[comm.get_rank()];

    auto bin_boundaries = pr::ndarray<Float, 1>::empty(queue, { rank_count + 1 }, alloc);

    auto trials = dal::array<Float>::empty(trials_count);
    auto host_trials = pr::ndview<Float, 1>::wrap_mutable(trials);

    const auto sample_block = propose_sample_block<Float>(queue, feature_count, sample_count);
    const auto candidate_block = propose_candidate_block<Float>(queue, feature_count, trials_count);
    auto distances = pr::ndarray<Float, 2>::empty(queue, { candidate_block, sample_block }, alloc);

    auto candidates = pr::ndarray<Float, 2>::empty(queue, { trials_count, feature_count }, alloc);
    auto candidate_indices = pr::ndarray<std::int64_t, 1>::empty(queue, { trials_count }, alloc);
    auto candidate_norms = pr::ndarray<Float, 1>::empty(queue, { trials_count }, alloc);
    auto potentials = pr::ndarray<Float, 1>::empty(queue, { trials_count }, alloc);

    auto sample_norms = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);
    auto sample_norms_fill = pr::fill(queue, sample_norms, Float(0), {});
    auto sample_norms_event =
        pr::reduce_by_rows(queue, samples, sample_norms, sum, square, { sample_norms_fill });

    auto dist_sq = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);
    auto closest = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);

    sycl::event last_event = first_sample(ctx,
                                          rng,
                                          full_count,
                                          centroids,
                                          boundaries,
                                          samples,
                                          potentials,
                                          dist_sq,
                                          candidate_norms,
                                          sample_norms,
                                          deps + sample_norms_event);

    // Obtain potential as a squared distance to the first sample
    Float curr_potential = potentials.at_device(queue, 0, { last_event });
    for (std::int64_t i = 1; i < cluster_count; ++i) {
        // Generates sequence (array) of random numbers on host
        generate_trials(rng, host_trials, curr_potential);

        auto closest_event = pr::copy(queue, closest, dist_sq, { last_event });

        auto device_trials = host_trials.to_device(queue, { last_event });
        auto search_event = find_indices(queue,
                                         comm,
                                         curr_offset,
                                         device_trials,
                                         dist_sq,
                                         bin_boundaries,
                                         candidate_indices,
                                         { last_event, closest_event });

        auto extract_event = extract_and_share_by_indices(ctx,
                                                          candidate_indices,
                                                          boundaries,
                                                          samples,
                                                          candidates,
                                                          { search_event });

        auto norms_event =
            pr::reduce_by_rows(queue, candidates, candidate_norms, sum, square, { extract_event });
        auto potential_event = compute_potential(ctx,
                                                 closest,
                                                 candidates,
                                                 samples,
                                                 distances,
                                                 potentials,
                                                 candidate_norms,
                                                 sample_norms,
                                                 { norms_event, closest_event });

        auto [valmin, argmin] = pr::argmin(queue, potentials, { potential_event });

        auto centroid = centroids.get_row_slice(i, i + 1);
        auto centroid_1d = centroid.template reshape<1>({ feature_count });
        auto chosen = candidates.get_row_slice(argmin, argmin + 1);
        auto copy_event = pr::copy(queue, centroid, chosen, {});

        auto chosen_norm = candidate_norms.get_slice(argmin, argmin + 1);
        auto chosen_event = compute_distances_to_cluster(queue,
                                                         centroid_1d,
                                                         samples,
                                                         dist_sq,
                                                         chosen_norm,
                                                         sample_norms,
                                                         { copy_event });
        auto min_event = min_number(queue, dist_sq, closest, { chosen_event });

        last_event = std::move(min_event);
        curr_potential = std::move(valmin);
    }

    sycl::event::wait_and_throw({ last_event });

    return compute_result<Task>{}.set_centroids(
        homogen_table::wrap(centroids_array, cluster_count, feature_count));
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

    return std::visit(
        [&](const auto& data) {
            return implementation<Method, Task>(ctx, params, data);
        },
        data_variant);
}

template struct compute_kernel_distr<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_distr<double, method::plus_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
