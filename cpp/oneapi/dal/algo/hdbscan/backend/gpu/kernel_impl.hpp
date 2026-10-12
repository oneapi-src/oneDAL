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

#pragma once

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/backend/atomic.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/search/copy_search_callback.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/// The brute-force path may use at most `1 / mrd_global_mem_divisor` of device global memory.
///
/// An oversubscribed `malloc_device` succeeds and then faults on first access instead of raising.
constexpr std::int64_t mrd_global_mem_divisor = 2;

/// Byte total the sizing helpers return on overflow; larger than any device limit.
constexpr std::int64_t mrd_bytes_saturated = dal::detail::limits<std::int64_t>::max();

/// Multiply two non-negative byte counts without signed overflow.
///
/// @param[in] lhs First factor
/// @param[in] rhs Second factor
///
/// @return `lhs * rhs`, or `mrd_bytes_saturated` when that product would not fit
inline std::int64_t mrd_mul_bytes(std::int64_t lhs, std::int64_t rhs) {
    ONEDAL_ASSERT(lhs >= 0);
    ONEDAL_ASSERT(rhs >= 0);
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    return (lhs > mrd_bytes_saturated / rhs) ? mrd_bytes_saturated : lhs * rhs;
}

/// Add two non-negative byte counts without signed overflow.
///
/// @param[in] lhs First summand
/// @param[in] rhs Second summand
///
/// @return `lhs + rhs`, or `mrd_bytes_saturated` when that sum would not fit
inline std::int64_t mrd_add_bytes(std::int64_t lhs, std::int64_t rhs) {
    ONEDAL_ASSERT(lhs >= 0);
    ONEDAL_ASSERT(rhs >= 0);
    return (lhs > mrd_bytes_saturated - rhs) ? mrd_bytes_saturated : lhs + rhs;
}

/// Decide whether the brute-force path's device buffers fit within the given limits.
///
/// The peak is the matrix plus the larger of the k-selection buffers and the MST edge arrays,
/// which are never live at the same time.
///
/// @param[in] row_count             Number of input rows `n`
/// @param[in] column_count          Number of input columns `d`
/// @param[in] element_size          Size in bytes of one distance-matrix element
/// @param[in] max_alloc_bytes       Device limit on a single allocation
/// @param[in] global_mem_bytes      Device global memory size
/// @param[in] min_samples           `k` of the core-distance k-selection
/// @param[in] kselect_scratch_bytes Device scratch of `kselect_by_rows`, from
///                                  `pr::kselect_by_rows_scratch_size`
///
/// @return `true` if the `n x n` matrix fits one allocation and the peak footprint fits the
///         global-memory budget
inline bool mrd_matrix_fits_on_device(std::int64_t row_count,
                                      std::int64_t column_count,
                                      std::int64_t element_size,
                                      std::int64_t max_alloc_bytes,
                                      std::int64_t global_mem_bytes,
                                      std::int64_t min_samples,
                                      std::int64_t kselect_scratch_bytes) {
    const std::int64_t row_bytes = mrd_mul_bytes(element_size, row_count);
    const std::int64_t matrix_bytes = mrd_mul_bytes(row_bytes, row_count);
    if (matrix_bytes > max_alloc_bytes) {
        return false;
    }

    // The input table and the core distances.
    const std::int64_t resident_bytes = mrd_mul_bytes(row_bytes, column_count + 1);

    // The `n x min_samples` k-selection output and its scratch.
    const std::int64_t selection_bytes =
        mrd_add_bytes(mrd_mul_bytes(row_bytes, min_samples), kselect_scratch_bytes);

    // The three MST edge arrays.
    const std::int64_t mst_bytes =
        mrd_mul_bytes(2 * static_cast<std::int64_t>(sizeof(std::int32_t)) + element_size,
                      row_count);

    const std::int64_t peak_bytes = mrd_add_bytes(mrd_add_bytes(matrix_bytes, resident_bytes),
                                                  std::max(selection_bytes, mst_bytes));
    return peak_bytes <= global_mem_bytes / mrd_global_mem_divisor;
}

/// Reject the row counts whose `n x n` MRD matrix does not fit on the device.
///
/// @tparam Float Floating-point type of the distance matrix
///
/// @param[in] queue        The SYCL queue whose device has to hold the matrix
/// @param[in] row_count    Number of input rows `n`
/// @param[in] column_count Number of input columns `d`
/// @param[in] min_samples  `k` of the core-distance k-selection
///
/// @throws domain_error if the matrix exceeds the single-allocation limit or if
///         the pipeline's peak footprint exceeds device global memory
template <typename Float>
inline void check_mrd_matrix_fits_on_device(sycl::queue& queue,
                                            std::int64_t row_count,
                                            std::int64_t column_count,
                                            std::int64_t min_samples) {
    constexpr std::int64_t element_size = static_cast<std::int64_t>(sizeof(Float));
    const std::int64_t max_alloc_bytes = bk::device_max_mem_alloc_size(queue);
    const std::int64_t global_mem_bytes = bk::device_global_mem_size(queue);

    // Size the matrix alone first: the scratch query needs an `n x n` shape to be representable.
    const bool matrix_fits = mrd_matrix_fits_on_device(row_count,
                                                       column_count,
                                                       element_size,
                                                       max_alloc_bytes,
                                                       global_mem_bytes,
                                                       min_samples,
                                                       0);
    const std::int64_t kselect_scratch_bytes =
        matrix_fits
            ? pr::kselect_by_rows_scratch_size<Float>(queue, { row_count, row_count }, min_samples)
            : 0;
    if (!matrix_fits || !mrd_matrix_fits_on_device(row_count,
                                                   column_count,
                                                   element_size,
                                                   max_alloc_bytes,
                                                   global_mem_bytes,
                                                   min_samples,
                                                   kselect_scratch_bytes)) {
        throw domain_error(
            dal::detail::error_messages::hdbscan_brute_force_matrix_does_not_fit_on_device());
    }
}

/// Match scikit-learn on zero-norm rows of a square cosine distance matrix.
///
/// The cosine primitive divides by the row norms, so a zero row yields NaN. scikit-learn
/// normalizes it to the zero vector instead, which puts it at distance 1 from every other row.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue The SYCL queue
/// @param[in]     data  Input data of size `n x d`
/// @param[in,out] dist  Cosine distance matrix of size `n x n`
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event fix_zero_norm_cosine(sycl::queue& queue,
                                        const pr::ndview<Float, 2>& data,
                                        pr::ndview<Float, 2>& dist,
                                        const bk::event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    pr::ndarray<Float, 1> norms;
    sycl::event norms_event;
    std::tie(norms, norms_event) = pr::compute_squared_l2_norms(queue, data, deps);
    const Float* const norms_ptr = norms.get_data();
    Float* const dist_ptr = dist.get_mutable_data();
    const std::int64_t stride = dist.get_leading_stride();

    auto event = bk::parallel_for_2d_by_row_blocks(
        queue,
        n,
        n,
        [=](std::int64_t i, std::int64_t j) {
            if (norms_ptr[i] == Float(0) || norms_ptr[j] == Float(0)) {
                dist_ptr[i * stride + j] = (i == j) ? Float(0) : Float(1);
            }
        },
        { norms_event });
    // `norms` is freed on return and `sycl::free` does not wait.
    event.wait_and_throw();
    return event;
}

/// Compute the full pairwise distance matrix; squared L2 for `euclidean`, final distances otherwise.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue  The SYCL queue
/// @param[in]  data   Input matrix of size `n x d`
/// @param[out] dist   Output matrix of size `n x n` (row-major)
/// @param[in]  metric Distance metric tag (`distance_metric`)
/// @param[in]  degree Minkowski degree (used only when `metric == minkowski`)
/// @param[in]  deps   Events that must complete before submission
///
/// @return Event signaling completion of the distance computation
template <typename Float>
inline sycl::event compute_distance_matrix(sycl::queue& queue,
                                           const pr::ndview<Float, 2>& data,
                                           pr::ndview<Float, 2>& dist,
                                           distance_metric metric,
                                           double degree,
                                           const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_distance_matrix, queue);

    const std::int64_t n = data.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);

    switch (metric) {
        case distance_metric::cosine: {
            pr::cosine_distance<Float> dist_op(queue);
            const auto dist_event = dist_op(data, data, dist, deps);
            return fix_zero_norm_cosine<Float>(queue, data, dist, { dist_event });
        }
        case distance_metric::manhattan: {
            pr::distance<Float, pr::lp_metric<Float>> dist_op(queue,
                                                              pr::lp_metric<Float>(Float(1)));
            return dist_op(data, data, dist, deps);
        }
        case distance_metric::minkowski: {
            pr::distance<Float, pr::lp_metric<Float>> dist_op(
                queue,
                pr::lp_metric<Float>(static_cast<Float>(degree)));
            return dist_op(data, data, dist, deps);
        }
        case distance_metric::chebyshev: {
            pr::chebyshev_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
        default: { // euclidean: squared L2, the root is taken later
            pr::squared_l2_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
    }
}

/// Compute per-point core distances as the `min_samples`-th smallest entry of each row.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  dist           Pairwise distance matrix of size `n x n`
/// @param[out] core_distances Per-point core distances, length `n`
/// @param[in]  min_samples    `k` used for the k-NN core-distance definition
/// @param[in]  row_count      Number of rows `n`
/// @param[in]  metric         Distance metric tag (controls the sqrt finalize)
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event compute_core_distances(sycl::queue& queue,
                                          const pr::ndview<Float, 2>& dist,
                                          pr::ndview<Float, 1>& core_distances,
                                          std::int64_t min_samples,
                                          std::int64_t row_count,
                                          distance_metric metric,
                                          const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == n);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= n);

    // The point itself counts as its first neighbor, as in scikit-learn.
    const std::int64_t k = min_samples;

    auto [ksel_vals, ksel_vals_event] =
        pr::ndarray<Float, 2>::zeros(queue, { n, k }, sycl::usm::alloc::device);

    pr::kselect_by_rows<Float> ksel(queue, { n, n }, k);
    bk::event_vector ksel_deps = deps;
    ksel_deps.push_back(ksel_vals_event);
    auto ksel_event = ksel(queue, dist, k, ksel_vals, ksel_deps);

    const Float* ksel_ptr = ksel_vals.get_data();
    Float* core_ptr = core_distances.get_mutable_data();
    const std::int64_t kk = k;
    const bool needs_sqrt = (metric == distance_metric::euclidean);

    auto extract_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ksel_event });
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float val = ksel_ptr[i * kk + (kk - 1)];
            core_ptr[i] = needs_sqrt ? sycl::sqrt(sycl::fmax(val, Float(0))) : val;
        });
    });

    // `ksel_vals` is freed on return and `sycl::free` does not wait for its readers.
    extract_event.wait_and_throw();
    return extract_event;
}

/// Compute per-point core distances with `pr::search_engine`, without the `n x n` matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  data           Input data of size `n x d`, device USM
/// @param[out] core_distances Per-point core distances, length `n`
/// @param[in]  min_samples    `k` used for the k-NN core-distance definition
/// @param[in]  block_size     Query rows per search block, in `[1, n]`
/// @param[in]  metric         Distance metric tag; cosine is not supported
/// @param[in]  degree         Minkowski degree
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event compute_core_distances_blocked(sycl::queue& queue,
                                                  const pr::ndview<Float, 2>& data,
                                                  pr::ndview<Float, 1>& core_distances,
                                                  std::int64_t min_samples,
                                                  std::int64_t block_size,
                                                  distance_metric metric,
                                                  double degree,
                                                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances_blocked, queue);

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t k = min_samples;
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(block_size >= 1 && block_size <= n);
    ONEDAL_ASSERT(k >= 1 && k <= n);
    ONEDAL_ASSERT(metric != distance_metric::cosine);

    auto knn_distances = pr::ndarray<Float, 2>::empty(queue, { n, k }, sycl::usm::alloc::device);
    pr::copy_callback<Float, false, true> callback(queue, block_size, {}, knn_distances);
    const std::int64_t train_block = pr::propose_train_block<Float>(queue, data.get_dimension(1));

    sycl::event search_event;
    const bool is_euclidean = (metric == distance_metric::euclidean);
    if (is_euclidean) {
        const pr::search_engine<Float, pr::squared_l2_distance<Float>> search(queue,
                                                                              data,
                                                                              train_block);
        search_event = search(data, callback, block_size, k, deps);
    }
    else if (metric == distance_metric::chebyshev) {
        const pr::chebyshev_distance<Float> dist(queue);
        const pr::search_engine<Float, pr::chebyshev_distance<Float>> search(queue,
                                                                             data,
                                                                             train_block,
                                                                             dist);
        search_event = search(data, callback, block_size, k, deps);
    }
    else {
        const Float p =
            metric == distance_metric::manhattan ? Float(1) : static_cast<Float>(degree);
        const pr::lp_distance<Float> dist(queue, pr::lp_metric<Float>(p));
        const pr::search_engine<Float, pr::lp_distance<Float>> search(queue,
                                                                      data,
                                                                      train_block,
                                                                      dist);
        search_event = search(data, callback, block_size, k, deps);
    }

    const Float* const knn_ptr = knn_distances.get_data();
    Float* const core_ptr = core_distances.get_mutable_data();
    auto extract_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(search_event);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float val = knn_ptr[i * k + (k - 1)];
            core_ptr[i] = is_euclidean ? sycl::sqrt(sycl::fmax(val, Float(0))) : val;
        });
    });

    // `knn_distances` is freed on return and `sycl::free` does not wait for its readers.
    extract_event.wait_and_throw();
    return extract_event;
}

/// Convert a distance matrix in place into `MRD(i, j) = max(core_i, core_j, dist(i, j)) / alpha`.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue          The SYCL queue
/// @param[in]     core_distances Per-point core distances, length `n` (unscaled)
/// @param[in,out] mrd_matrix     Distance matrix `n x n` (squared L2 for euclidean), overwritten
///                               with MRD values
/// @param[in]     metric         Distance metric tag (controls the sqrt finalize)
/// @param[in]     alpha          Robust single-linkage scaling factor; divides the whole MRD
/// @param[in]     deps           Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event compute_mrd_matrix(sycl::queue& queue,
                                      const pr::ndview<Float, 1>& core_distances,
                                      pr::ndview<Float, 2>& mrd_matrix,
                                      distance_metric metric,
                                      double alpha,
                                      const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_mrd_matrix, queue);

    const std::int64_t n = core_distances.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    const Float* core_ptr = core_distances.get_data();
    Float* mrd_ptr = mrd_matrix.get_mutable_data();
    const bool needs_sqrt = (metric == distance_metric::euclidean);
    const Float inv_alpha = static_cast<Float>(1.0 / alpha);

    return bk::parallel_for_2d_by_row_blocks(
        queue,
        n,
        n,
        [=](std::int64_t i, std::int64_t j) {
            // Euclidean entries hold squared L2; the other metrics hold final distances.
            const Float d = needs_sqrt ? sycl::sqrt(sycl::fmax(mrd_ptr[i * n + j], Float(0)))
                                       : mrd_ptr[i * n + j];
            // scikit-learn's brute path scales the core distances by 1/alpha as well.
            const Float cd_i = core_ptr[i] * inv_alpha;
            const Float cd_j = core_ptr[j] * inv_alpha;
            mrd_ptr[i * n + j] = sycl::fmax(sycl::fmax(cd_i, cd_j), d * inv_alpha);
        },
        deps);
}

/// Find each point's nearest neighbor in another component from a precomputed MRD matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue           The SYCL queue
/// @param[in]  mrd_ptr         Precomputed MRD matrix of size `n x n`
/// @param[in]  comp_ptr        Per-point component id, length `n`
/// @param[out] pt_best_mrd_ptr Per-point best MRD, length `n`
/// @param[out] pt_best_idx_ptr Per-point best different-component column index, length `n`
/// @param[in]  n               Number of points
/// @param[in]  deps            Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_find_nearest_mrd(sycl::queue& queue,
                                            const Float* mrd_ptr,
                                            const std::int32_t* comp_ptr,
                                            Float* pt_best_mrd_ptr,
                                            std::int32_t* pt_best_idx_ptr,
                                            std::int64_t n,
                                            const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t my_comp = comp_ptr[i];
            const Float* row = mrd_ptr + i * n;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_ptr[j] != my_comp && row[j] < best_m) {
                    best_m = row[j];
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd_ptr[i] = best_m;
            pt_best_idx_ptr[i] = best_j;
        });
    });
}

/// Find each point's nearest neighbor in another component, computing the distances on the fly.
///
/// Uses `MRD(i, j) = max(core_i, core_j, dist(i, j) * inv_alpha)`, so no `n x n` matrix is held.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue           The SYCL queue
/// @param[in]  data_ptr        Row-major input buffer, size `n x col_count`
/// @param[in]  col_count       Number of features
/// @param[in]  core_ptr        Per-point core distances, length `n` (unscaled)
/// @param[in]  comp_ptr        Per-point component id, length `n`
/// @param[out] pt_best_mrd_ptr Per-point best MRD, length `n`
/// @param[out] pt_best_idx_ptr Per-point best different-component point index, length `n`
/// @param[in]  n               Number of points
/// @param[in]  metric_id       0=euclidean, 1=manhattan, 2=minkowski, 3=chebyshev
/// @param[in]  degree          Minkowski degree (used only when `metric_id == 2`)
/// @param[in]  inv_alpha       `1.0 / alpha`, applied only to dist(i,j) inside MRD
/// @param[in]  deps            Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_find_nearest_otf(sycl::queue& queue,
                                            const Float* data_ptr,
                                            std::int64_t col_count,
                                            const Float* core_ptr,
                                            const std::int32_t* comp_ptr,
                                            Float* pt_best_mrd_ptr,
                                            std::int32_t* pt_best_idx_ptr,
                                            std::int64_t n,
                                            std::int32_t metric_id,
                                            Float degree,
                                            Float inv_alpha,
                                            const bk::event_vector& deps) {
    const std::int64_t d = col_count;
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t my_comp = comp_ptr[i];
            const Float my_core = core_ptr[i];
            const Float* xi = data_ptr + i * d;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_ptr[j] == my_comp)
                    continue;
                const Float* xj = data_ptr + j * d;
                Float dist = Float(0);
                if (metric_id == 1) { // manhattan
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = xi[dd] - xj[dd];
                        dist += sycl::fabs(diff);
                    }
                }
                else if (metric_id == 2) { // minkowski
                    Float sum = Float(0);
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = sycl::fabs(xi[dd] - xj[dd]);
                        sum += sycl::pow(diff, degree);
                    }
                    dist = sycl::pow(sum, Float(1) / degree);
                }
                else if (metric_id == 3) { // chebyshev
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = sycl::fabs(xi[dd] - xj[dd]);
                        if (diff > dist)
                            dist = diff;
                    }
                }
                else { // euclidean
                    Float sum = Float(0);
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = xi[dd] - xj[dd];
                        sum += diff * diff;
                    }
                    dist = sycl::sqrt(sycl::fmax(sum, Float(0)));
                }
                Float mrd = sycl::fmax(sycl::fmax(my_core, core_ptr[j]), dist * inv_alpha);
                if (mrd < best_m) {
                    best_m = mrd;
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd_ptr[i] = best_m;
            pt_best_idx_ptr[i] = best_j;
        });
    });
}

/// Reduce per-point bests to per-component bests, then merge components and append the MST edges.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue              The SYCL queue
/// @param[in,out] comp_ptr           Per-point component id, length `n` (re-read after compress)
/// @param[in,out] uf_parent_ptr      Union-find parent array, length `n`
/// @param[in,out] uf_rank_ptr        Union-find rank array, length `n`
/// @param[in]     pt_best_mrd_ptr    Per-point best MRD from the find phase
/// @param[in]     pt_best_idx_ptr    Per-point best different-component index from the find phase
/// @param[out]    comp_best_mrd_ptr  Scratch: per-component best MRD, length `n`
/// @param[out]    comp_best_from_ptr Scratch: per-component best `from` index, length `n`
/// @param[out]    mst_from_ptr       Output MST `from` endpoints
/// @param[out]    mst_to_ptr         Output MST `to` endpoints
/// @param[out]    mst_weight_ptr     Output MST weights
/// @param[in,out] edges_added_ptr    Single-element counter of MST edges added so far
/// @param[in,out] num_comp_ptr       Single-element counter of remaining components
/// @param[in]     n                  Number of points
/// @param[in]     deps               Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_merge_components(sycl::queue& queue,
                                            std::int32_t* comp_ptr,
                                            std::int32_t* uf_parent_ptr,
                                            std::int32_t* uf_rank_ptr,
                                            const Float* pt_best_mrd_ptr,
                                            const std::int32_t* pt_best_idx_ptr,
                                            Float* comp_best_mrd_ptr,
                                            std::int32_t* comp_best_from_ptr,
                                            std::int32_t* mst_from_ptr,
                                            std::int32_t* mst_to_ptr,
                                            Float* mst_weight_ptr,
                                            std::int32_t* edges_added_ptr,
                                            std::int32_t* num_comp_ptr,
                                            std::int64_t n,
                                            const bk::event_vector& deps) {
    const Float inf = std::numeric_limits<Float>::max();
    constexpr std::int32_t no_point = std::numeric_limits<std::int32_t>::max();
    const auto range = sycl::range<1>(n);

    auto reset_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> c) {
            comp_best_mrd_ptr[c] = inf;
            comp_best_from_ptr[c] = no_point;
        });
    });

    // Two atomic passes reproduce the serial reduction exactly: the smallest MRD per component,
    // then the smallest point index among the points that reach it.
    auto min_mrd_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(reset_event);
        h.parallel_for(range, [=](sycl::id<1> i) {
            if (pt_best_idx_ptr[i] >= 0) {
                bk::atomic_global_min(comp_best_mrd_ptr + comp_ptr[i], pt_best_mrd_ptr[i]);
            }
        });
    });

    auto min_idx_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(min_mrd_event);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            const std::int32_t c = comp_ptr[i];
            const Float mrd = pt_best_mrd_ptr[i];
            if (pt_best_idx_ptr[i] >= 0 && mrd < inf && mrd == comp_best_mrd_ptr[c]) {
                bk::atomic_global_min(comp_best_from_ptr + c, i);
            }
        });
    });

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(min_idx_event);
        h.single_task([=]() {
            auto uf_find = [&](std::int32_t x) -> std::int32_t {
                while (uf_parent_ptr[x] != x) {
                    uf_parent_ptr[x] = uf_parent_ptr[uf_parent_ptr[x]];
                    x = uf_parent_ptr[x];
                }
                return x;
            };
            auto uf_union = [&](std::int32_t rx, std::int32_t ry) {
                if (uf_rank_ptr[rx] < uf_rank_ptr[ry])
                    uf_parent_ptr[rx] = ry;
                else if (uf_rank_ptr[rx] > uf_rank_ptr[ry])
                    uf_parent_ptr[ry] = rx;
                else {
                    uf_parent_ptr[ry] = rx;
                    uf_rank_ptr[rx]++;
                }
            };

            std::int32_t ea = edges_added_ptr[0];
            std::int32_t added = 0;
            for (std::int64_t c = 0; c < n; c++) {
                const std::int32_t u = comp_best_from_ptr[c];
                if (u == no_point)
                    continue;
                const std::int32_t v = pt_best_idx_ptr[u];
                const std::int32_t ru = uf_find(u), rv = uf_find(v);
                if (ru == rv)
                    continue;
                mst_from_ptr[ea] = u;
                mst_to_ptr[ea] = v;
                mst_weight_ptr[ea] = comp_best_mrd_ptr[c];
                ea++;
                added++;
                uf_union(ru, rv);
            }
            edges_added_ptr[0] = ea;
            num_comp_ptr[0] -= added;
        });
    });
}

/// Set `comp[i] = find(i)` for every point after a Boruvka merge round.
///
/// @param[in]  queue         The SYCL queue
/// @param[out] comp_ptr      Per-point component id, length `n` (overwritten)
/// @param[in]  uf_parent_ptr Union-find parent array, length `n`
/// @param[in]  n             Number of points
/// @param[in]  deps          Events that must complete before submission
///
/// @return Event signaling completion
inline sycl::event boruvka_compress_components(sycl::queue& queue,
                                               std::int32_t* comp_ptr,
                                               const std::int32_t* uf_parent_ptr,
                                               std::int64_t n,
                                               const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            std::int32_t x = static_cast<std::int32_t>(idx[0]);
            while (uf_parent_ptr[x] != x)
                x = uf_parent_ptr[x];
            comp_ptr[idx[0]] = x;
        });
    });
}

/// Build the MST under MRD with Boruvka's algorithm from a precomputed MRD matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue       The SYCL queue
/// @param[in]  mrd_matrix  Precomputed MRD matrix of size `n x n`
/// @param[out] mst_from    Output MST `from` endpoints, length `n - 1`
/// @param[out] mst_to      Output MST `to` endpoints, length `n - 1`
/// @param[out] mst_weights Output MST weights, length `n - 1`
/// @param[in]  row_count   Number of points `n`
/// @param[in]  deps        Events that must complete before submission
///
/// @return Event signaling completion of the final Boruvka round
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline sycl::event build_mst(sycl::queue& queue,
                             const pr::ndview<Float, 2>& mrd_matrix,
                             pr::ndview<std::int32_t, 1>& mst_from,
                             pr::ndview<std::int32_t, 1>& mst_to,
                             pr::ndview<Float, 1>& mst_weights,
                             std::int64_t row_count,
                             const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 1);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    // Allocate working arrays on device
    auto [comp, comp_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_parent, uf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_rank, ur_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_mrd, pbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_idx, pbi_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_mrd, cbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_from, cbf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(init_deps.end(),
                     { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, ea_ev, nc_ev });

    // Initialize comp[i] = i, uf_parent[i] = i, num_comp = n
    auto init_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(init_deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            comp_ptr[i] = i;
            uf_parent_ptr[i] = i;
            if (i == 0)
                num_comp_ptr[0] = static_cast<std::int32_t>(n);
        });
    });

    const Float* mrd_ptr = mrd_matrix.get_data();
    const std::int32_t max_rounds = 64;
    sycl::event last_event = init_event;
    init_event.wait_and_throw();

    bool connected = (n <= 1);
    for (std::int32_t round = 0; round < max_rounds; round++) {
        // Step A: nearest neighbor in another component
        auto find_event = boruvka_find_nearest_mrd<Float>(queue,
                                                          mrd_ptr,
                                                          comp_ptr,
                                                          pt_best_mrd.get_mutable_data(),
                                                          pt_best_idx.get_mutable_data(),
                                                          n,
                                                          { last_event });
        find_event.wait_and_throw();

        // Step B: reduce and merge
        auto merge_event = boruvka_merge_components<Float>(queue,
                                                           comp_ptr,
                                                           uf_parent_ptr,
                                                           uf_rank_ptr,
                                                           pt_best_mrd.get_data(),
                                                           pt_best_idx.get_data(),
                                                           comp_best_mrd.get_mutable_data(),
                                                           comp_best_from.get_mutable_data(),
                                                           mst_from.get_mutable_data(),
                                                           mst_to.get_mutable_data(),
                                                           mst_weights.get_mutable_data(),
                                                           edges_added_arr.get_mutable_data(),
                                                           num_comp_arr.get_mutable_data(),
                                                           n,
                                                           { find_event });
        merge_event.wait_and_throw();

        // Step C: path compression
        auto compress_event =
            boruvka_compress_components(queue, comp_ptr, uf_parent_ptr, n, { merge_event });
        compress_event.wait_and_throw();

        // Check termination
        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        const std::int32_t nc = num_comp_host.get_data()[0];
        last_event = compress_event;
        if (nc <= 1) {
            connected = true;
            break;
        }
    }

    // Only non-finite input leaves components that no round can join.
    if (!connected) {
        throw domain_error(dal::detail::error_messages::hdbscan_input_data_is_not_finite());
    }
    return last_event;
}

/// Build the MST under MRD with Boruvka's algorithm, computing the distances on the fly.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  data           Row-major input buffer of size `n x col_count`
/// @param[in]  core_distances Per-point core distances, length `n`
/// @param[out] mst_from       Output MST `from` endpoints, length `n - 1`
/// @param[out] mst_to         Output MST `to` endpoints, length `n - 1`
/// @param[out] mst_weights    Output MST weights, length `n - 1`
/// @param[in]  row_count      Number of points `n`
/// @param[in]  col_count      Number of features `d`
/// @param[in]  metric         Distance metric tag
/// @param[in]  degree         Minkowski degree (used only when `metric == minkowski`)
/// @param[in]  alpha          Robust single-linkage scaling factor of `dist(i, j)` inside MRD
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion of the final Boruvka round
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline sycl::event build_mst_otf(sycl::queue& queue,
                                 const pr::ndview<Float, 2>& data,
                                 const pr::ndview<Float, 1>& core_distances,
                                 pr::ndview<std::int32_t, 1>& mst_from,
                                 pr::ndview<std::int32_t, 1>& mst_to,
                                 pr::ndview<Float, 1>& mst_weights,
                                 std::int64_t row_count,
                                 std::int64_t col_count,
                                 distance_metric metric,
                                 double degree,
                                 double alpha,
                                 const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst_otf, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 1);

    std::int32_t metric_id = 0; // euclidean
    if (metric == distance_metric::manhattan)
        metric_id = 1;
    else if (metric == distance_metric::minkowski)
        metric_id = 2;
    else if (metric == distance_metric::chebyshev)
        metric_id = 3;

    // Allocate working arrays on device
    auto [comp, comp_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_parent, uf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_rank, ur_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_mrd, pbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_idx, pbi_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_mrd, cbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_from, cbf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(init_deps.end(),
                     { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, ea_ev, nc_ev });

    auto init_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(init_deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            comp_ptr[i] = i;
            uf_parent_ptr[i] = i;
            if (i == 0)
                num_comp_ptr[0] = static_cast<std::int32_t>(n);
        });
    });

    const Float* data_ptr = data.get_data();
    const Float* core_ptr = core_distances.get_data();
    const Float deg_f = static_cast<Float>(degree);
    const Float inv_alpha = static_cast<Float>(1.0 / alpha);
    const std::int32_t max_rounds = 64;
    sycl::event last_event = init_event;
    init_event.wait_and_throw();

    bool connected = (n <= 1);
    for (std::int32_t round = 0; round < max_rounds; round++) {
        auto find_event = boruvka_find_nearest_otf<Float>(queue,
                                                          data_ptr,
                                                          col_count,
                                                          core_ptr,
                                                          comp_ptr,
                                                          pt_best_mrd.get_mutable_data(),
                                                          pt_best_idx.get_mutable_data(),
                                                          n,
                                                          metric_id,
                                                          deg_f,
                                                          inv_alpha,
                                                          { last_event });
        find_event.wait_and_throw();

        auto merge_event = boruvka_merge_components<Float>(queue,
                                                           comp_ptr,
                                                           uf_parent_ptr,
                                                           uf_rank_ptr,
                                                           pt_best_mrd.get_data(),
                                                           pt_best_idx.get_data(),
                                                           comp_best_mrd.get_mutable_data(),
                                                           comp_best_from.get_mutable_data(),
                                                           mst_from.get_mutable_data(),
                                                           mst_to.get_mutable_data(),
                                                           mst_weights.get_mutable_data(),
                                                           edges_added_arr.get_mutable_data(),
                                                           num_comp_arr.get_mutable_data(),
                                                           n,
                                                           { find_event });
        merge_event.wait_and_throw();

        auto compress_event =
            boruvka_compress_components(queue, comp_ptr, uf_parent_ptr, n, { merge_event });
        compress_event.wait_and_throw();

        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        const std::int32_t nc = num_comp_host.get_data()[0];
        last_event = compress_event;
        if (nc <= 1) {
            connected = true;
            break;
        }
    }

    // Only non-finite input leaves components that no round can join.
    if (!connected) {
        throw domain_error(dal::detail::error_messages::hdbscan_input_data_is_not_finite());
    }
    return last_event;
}

#endif

} // namespace oneapi::dal::hdbscan::backend
