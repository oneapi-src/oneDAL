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

/// @brief Handles user input of trial count. In case of
///        positive number returns it, in case of -1 computes
///        value on its own
///
/// @cite https://github.com/scikit-learn/scikit-learn/blob/1.2.X/sklearn/cluster/_kmeans.py#L207
///
/// @param trial_count[in]   User defined number of wanted trials
/// @param cluster_count[in] Number of centroids requested by user
/// @return                  Number of trials, positive integer number
std::int64_t fix_trials_count(std::int64_t trial_count, std::int64_t cluster_count) {
    ONEDAL_ASSERT(trial_count > 0 || trial_count == -1);

    const auto additional = std::log(cluster_count);
    const auto proposed = 2 + std::int64_t(additional);
    auto result = (trial_count == -1) ? proposed : trial_count;

    ONEDAL_ASSERT(result > 0l);

    return result;
}

/// @brief Adds number to the array
///
/// @tparam Type Type of values to add
///
/// @param[in] queue      SYCL queue to run kernels on
/// @param[in] value      Value to add to the array
/// @param[in, out] array Array to add number to
/// @param[in] deps       Dependencies for this kernel
/// @return               SYCL event with progress of the kernel
template <typename Type>
sycl::event add_number(sycl::queue& queue,
                       const pr::ndview<Type, 1>& value,
                       pr::ndview<Type, 1>& array,
                       const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(value.has_data());
    ONEDAL_ASSERT(array.has_mutable_data());

    const Type* const val_ptr = value.get_data();
    ONEDAL_ASSERT(value.get_count() == std::int64_t(1));

    const auto kernel = [](Type prev, const Type* val_ptr) -> Type {
        return prev + *val_ptr;
    };

    return element_wise(queue, kernel, array, val_ptr, array, deps);
}

/// Can combine 1D & 2D arrays

/// @brief Combines minimum numbers by the last axis
///
/// @tparam Type Type of data to handle, should be common
/// @tparam ax1  Number of axis in output array
/// @tparam ax2  Number of axis in the second array
///
/// @param[in] queue      Represents device to run on
/// @param[in, out] array One of operands, mutable
/// @param[in] minimum    The second operant
/// @param[in] deps       SYCL dependencies for this kernel
/// @return               Event to allow asynchroous usage
///                       of the kernel results
template <typename Type, std::int64_t ax1, std::int64_t ax2>
sycl::event min_number(sycl::queue& queue,
                       pr::ndview<Type, ax1>& array,
                       const pr::ndview<Type, ax2> minimum,
                       const bk::event_vector& deps = {}) {
    constexpr sycl::minimum<Type> kernel{};
    ONEDAL_ASSERT(array.has_mutable_data());
    return element_wise(queue, kernel, array, minimum, array, deps);
}

/// @brief Finds span between two values in the `offsets`
///        array by using standrd library functionality
///        Can utilize `searchsorted` function but it is
///        complicated due to the data location
///
/// @tparam Type
///
/// @param offsets[in] Boundaries from different bins
///                    (usually from cluster ranks)
/// @param value[in]   Value to find location in bins
/// @return            Returns bin index value fit in
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

/// @brief Computes boundaries for bins from different ranks
///
/// @tparam Comm     Communicator entity that represents cluster
/// @tparam Type     Type of values to collect
///
/// @param comm[in]  Communicator entity
/// @param local[in] Values to collect from different ranks
/// @return          Basically cumulative sum of values from
///                  different ranks
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
        // comm.bcast() is sporadically failing by timeout
        // TODO: Need to fix comm.bcast() and replace comm.allreduce() with comm.bcast()
        if (rank != target) {
            pr::fill(queue, place, Float(0), new_deps).wait_and_throw();
        }
        auto wrap_place =
            array<Float>::wrap(queue, place.get_mutable_data(), place.get_count(), new_deps);
        comm.allreduce(wrap_place).wait();
    }

    return bk::wait_or_pass(new_deps);
}

/// @brief The version that extracts samples all at once
template <typename Float, pr::ndorder order>
sycl::event extract_and_share_by_indices_narrow(const bk::context_gpu& ctx,
                                                const pr::ndview<std::int64_t, 1>& indices,
                                                const dal::array<std::int64_t>& offsets,
                                                const pr::ndview<Float, 2, order>& input,
                                                pr::ndview<Float, 2>& candidates,
                                                const bk::event_vector& deps = {}) {
    const auto candidate_count = candidates.get_dimension(0);
    const auto feature_count = candidates.get_dimension(1);
    ONEDAL_ASSERT(candidate_count == indices.get_count());
    auto* candidate_ptr = candidates.get_mutable_data();
    const auto* indices_ptr = indices.get_data();

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();
    ONEDAL_ASSERT(comm.get_rank_count() + 1 == offsets.get_count());

    const auto rank = comm.get_rank();
    const auto lower_bound = offsets[rank];
    const auto upper_bound = offsets[rank + 1];

    const auto input_indexer = make_ndindexer(input);

    auto range = bk::make_range_2d(candidate_count, feature_count);

    auto extract_local = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto index = indices_ptr[idx[0]];
            const auto local_index = index - lower_bound;
            auto* output = candidate_ptr + feature_count * idx[0] + idx[1];
            const auto handle = (lower_bound <= index) && (index < upper_bound);
            *output = handle ? input_indexer.at(local_index, idx[1]) : Float(0);
        });
    });

    if (comm.get_rank_count() > 1) {
        sycl::event::wait_and_throw({ extract_local });
        auto array = dal::array<Float>::wrap(queue,
                                             candidates.get_mutable_data(), //
                                             candidate_count * feature_count);

        comm.allreduce(array).wait();
    }

    return extract_local;
}

/// @brief The version that extracts samples one by one calling
///        `extract_and_share_by_index` sequentially
template <typename Float, pr::ndorder order>
sycl::event extract_and_share_by_indices_wide(const bk::context_gpu& ctx,
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

/// @brief Produces the same set of centroids on
///        all ranks by global indices
///
/// @todo  Optimize by extracting samples all at once
///        and combining them using allreduce
///
/// @tparam Float Type of data to handle
/// @tparam order Data layout in the original dataset
///
/// @param ctx[in]         SYCL queue and communicator as a single object
/// @param indices[in]     Global indices to select across different ranks
/// @param offsets[in]     Cumulative sum of number of samples on ranks
/// @param input[in]       Original shard of dataset on this rank
/// @param candidates[out] Selected across ranks samples
/// @param deps[in]        Vector of SYCL events
/// @return                SYCL event of the last event
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

    constexpr std::int64_t type_size = sizeof(Float);
    constexpr std::int64_t threshold_count = 131'072l;

    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, candidate_count, feature_count);
    const auto element_count = candidate_count * feature_count;

    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, type_size, element_count);
    const bool use_wide = (candidate_count > feature_count) //
                          && (type_size * element_count > threshold_count);

    if (use_wide) {
        return extract_and_share_by_indices_wide(ctx, indices, offsets, input, candidates, deps);
    }
    else {
        return extract_and_share_by_indices_narrow(ctx, indices, offsets, input, candidates, deps);
    }
}

template <typename Comm, typename Type>
Type get_local_offset(Comm& comm, const Type& value, const dal::array<Type>& temp) {
    ONEDAL_ASSERT(temp.has_mutable_data());
    const auto rank_count = comm.get_rank_count();
    ONEDAL_ASSERT(rank_count == temp.get_count());

    if (rank_count > 1) {
        constexpr Type zero(0);
        comm.allgather(value, temp).wait();
        return std::accumulate(bk::begin(temp), bk::end(temp), zero);
    }
    else {
        return value;
    }
}

/// @brief Computes sum of elements before current rank
///
/// @tparam Comm Communicator type
/// @tparam Type Type of collected values
///
/// @param comm[in]  Communicator entity
/// @param value[in] Value on this rank to collect
/// @return          Sum of values on lower ranks
template <typename Comm, typename Type>
Type get_local_offset(Comm& comm, const Type& value) {
    const auto rank_count = comm.get_rank_count();
    if (rank_count == 1)
        return value;
    auto temp = dal::array<Type>::empty(rank_count);
    return get_local_offset(comm, value, temp);
}

/// @brief Creates an array of random numbers in range [0, pot)
///
/// @tparam Generator Type of random number generator, usually std::mt19937
/// @tparam Float     Type of output values, the same with data type of algorithm
/// @tparam GenFloat  Type of floating point that is used for generatioon
///
/// @param rng[in]    Random Number Generator entity
/// @param trls[out]  Output array, should have correct size
/// @param pot[in]    Potential value, that limits upper bound
template <typename Generator, typename Float, typename GenFloat = float>
void generate_trials(Generator& rng, pr::ndview<Float, 1>& trls, double pot) {
    std::uniform_real_distribution<GenFloat> finalize(0.0, pot);
    const auto gen = [&]() -> Float {
        return finalize(rng);
    };
    std::generate(bk::begin(trls), bk::end(trls), gen);
}

/// @brief Proposes          Number of samples in slice to process at once
///
/// @tparam Float            We need to take type into account to find
///                          the best block size
///
/// @param queue[in]         Current queut to take current device into account
/// @param feature_count[in] Number of features in the dataset
/// @param sample_count[in]  Number of samples in the original dataset
/// @return                  "Optimal" number of samples in the sample block
template <typename Float>
std::int64_t propose_sample_block(const sycl::queue& queue,
                                  std::int64_t feature_count,
                                  std::int64_t sample_count) {
    return std::min<std::int64_t>(sample_count, 16'384l);
}

/// @brief Proposes          Number of candidates in slice to process at once
///
/// @tparam Float            We need to take type into account to find
///                          the best block size
///
/// @param queue[in]         Current queut to take current device into account
/// @param feature_count[in] Number of features in the dataset
/// @param sample_count[in]  Number of samples in the original dataset
/// @return                  "Optimal" number of candidates in the candidate block
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
        const auto cd_dist = cd_last - cd_first;
        ONEDAL_ASSERT(cd_dist > 0);

        const auto candidate_slice = candidates.get_row_slice(cd_first, cd_last);
        const auto candidate_norms_slice = candidate_norms.get_slice(cd_first, cd_last);

        auto dist_cd_slice = distances.get_row_slice(0l, cd_dist);
        auto pot_cd_slice = potential.get_slice(cd_first, cd_last);

        for (std::int64_t sp_id = 0; sp_id < sample_blocking.get_block_count(); ++sp_id) {
            const auto sp_first = sample_blocking.get_block_start_index(sp_id);
            const auto sp_last = sample_blocking.get_block_end_index(sp_id);
            const auto sp_dist = sp_last - sp_first;
            ONEDAL_ASSERT(sp_dist > 0);

            const auto closest_slice = closest.get_slice(sp_first, sp_last);
            const auto sample_slice = samples.get_row_slice(sp_first, sp_last);
            const auto sample_norms_slice = sample_norms.get_slice(sp_first, sp_last);

            auto dist_sp_slice = dist_cd_slice.get_col_slice(0l, sp_dist);

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
                                            /* deps =*/{ min_event },
                                            /*override_init =*/false);
        }
    }

    return last_event;
}

/// @brief Computes squared distances, compares such distances with
///        ones computed in the previous step, computes potentials
///
/// @tparam Float              Floating point for computations
/// @tparam order              Data layout of the input dataset
///
/// @param ctx[in]             Conbines SYCL event with communicator
/// @param closest[in]         Squared distances on previous iteration
/// @param candidates[in]      Candidate centroids
/// @param samples[in]         Original dataset
/// @param distances           Scratchpad place to compute distances here
/// @param potential[out]      Potentials of candidate centroids
/// @param candidate_norms[in] L2 norms of candidates
/// @param sample_norms[in]    L2 norms of the original dataset
/// @param deps[in]            Vector of SYCL dependencies
/// @return                    Last event in sequence
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

/// @brief Computes distances from local samples to a single centroid
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

/// @brief Somewhat complex function with many responsibilities
///        In general it: generates index, extracts it,
///        computes potential and squared distances
///
/// @tparam Generator           Random number generator to find index
/// @tparam Float               Floating point type to perform computations
/// @tparam order               Data layout of the dataset
///
/// @param[in]  ctx             Both queue and communicator
/// @param[in]  rng             Random number generator state
/// @param[in]  full_count      Number of samples in dataset
/// @param[out] centroids       Output with a single centroid
/// @param[in]  boundaries      Index offsets for different ranks
/// @param[in]  samples         Input dataset
/// @param[out] potential       Potential of the first sample
/// @param[out] distances       Squared distances of the sample
/// @param[out] candidates_norm L2 norm of the first sample
/// @param[out] samples_norm    L2 norms of the dataset
/// @param[in]  deps            Dependencies of this kernel
/// @return                     SYCL event that represents the last stage progress
template <typename Generator, typename Float, pr::ndorder order>
sycl::event first_sample(const bk::context_gpu& ctx,
                         Generator& rng,
                         const std::int64_t full_count,
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

/// @brief Checks and adjusts indices in case they belong to different ranks
///
/// @tparam Communicator Represents communicator library interface
/// @tparam Float        Floating point type to perform compuattions
/// @tparam Index        Integer type that represents indices of samples,
///                      usually std::int64_t
///
/// @param[in]  queue    SYCL queue with the device to run on
/// @param[in]  comm     Communicator's library interface
/// @param[in]  offset   First index on this rank, computed on host
/// @param[in]  points   Randomly generated points to find bin for
/// @param[in]  bounds   Boundaries of bins for different ranks
/// @param[out] indices  Proposed indices for point values
/// @param[in]  deps     Dependencies of this kernel
/// @return              SYCL event that represents the last stage progress
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

    if (rank_count > 1) {
        ONEDAL_PROFILER_TASK(fix_indices, queue);
        sycl::event::wait_and_throw({ event });

        const auto ids_arr = array<std::int64_t>::wrap(queue, //
                                                       indices.get_mutable_data(),
                                                       indices.get_count(),
                                                       { event });

        comm.allreduce(ids_arr, dal::preview::spmd::reduce_op::max).wait();

        return sycl::event{};
    }
    else {
        return event;
    }
}

/// @brief Computes cumulative sum, adjusts it in case of distributed
///        execution, finds the best location for randomly generated
///        values in this sequence
///
/// @tparam Communicator Represents communicator library interface
/// @tparam Float        Floating point type to perform compuattions
/// @tparam Index        Integer type that represents indices of samples,
///                      usually std::int64_t
///
/// @param[in]  queue    SYCL queue with the device to run on
/// @param[in]  comm     Communicator's library interface
/// @param[in]  offset   First index on this rank, computed on host
/// @param[in]  points   Randomly generated points to find bin for
/// @param[in]  values   Squared distances to samples computed on previous stage
/// @param[in]  bounds   Boundaries of bins for different ranks
/// @param[out] indices  Proposed indices for point values
/// @param[in]  deps     Dependencies of this kernel
/// @return              SYCL event that represent the alst stage progress
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

/// @brief The actual body of the KMeans++ init
/// Bird eye view on the function:
///     0) Does preparation work i.e. allocations
///     1) Finds first sample blindly according to RNG value
///     2) Computes potential and squared distances
/// For all other samples:
///     3) Generates several random numbers in range [0, potential)
///     4) Finds indices and extracts all centroid candidates
///     5) Computes potential for all candidates
///     6) Select the best candidate with the smallest potential
///     7) Recomputes squared distance (to avoid storing large array)
///
/// @tparam Method   Dummy template parameter to avoid symbol collisions
/// @tparam Task     Dummy template parameter to avoid symbol collisions
/// @tparam Float    Floating-point type used to perform computations
/// @tparam order    Input matrix data layout, is used by std::visit
///
/// @param[in]  ctx     The SYCL queue and communicator packed together
/// @param[in]  params  Description of the algorithm including number of trials, etc.
/// @param[in]  samples The [n x p] input dataset with specified fp type and data layout
/// @param[in]  deps    The vector of `sycl::event`s that represents list of SYCL dependencies
/// @return             Standard algorithm result with centroids and fails
///                     in case of template parameters
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
template struct compute_kernel_distr<float, method::plus_plus_csr, task::init>;
template struct compute_kernel_distr<double, method::plus_plus_csr, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
