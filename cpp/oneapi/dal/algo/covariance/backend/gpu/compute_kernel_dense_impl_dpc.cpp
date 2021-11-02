/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::covariance::backend {

#define ASSERT_IF(enable_condition, condition)                   \
    do {                                                         \
        if constexpr (check_mask_flag(enable_condition, List)) { \
            ONEDAL_ASSERT(condition);                            \
        }                                                        \
    } while (0)

#define DECLSET_IF(type, var, cond, value)       \
    type var = nullptr;                          \
    if constexpr (check_mask_flag(cond, List)) { \
        var = value;                             \
    }

#define SET_IF(var, cond, value)                 \
    if constexpr (check_mask_flag(cond, List)) { \
        var = value;                             \
    }

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

using comm_t = bk::communicator<spmd::device_memory_access::usm>;
using dal::backend::context_gpu;
using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, cov_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_row_block_count(std::int64_t row_count) {
    ONEDAL_ASSERT(row_count > 0);
    // TODO optimize the approach for row_block_count calculating

    std::int64_t row_block_count = 128;
    if (row_count < 5000)
        row_block_count = 1;
    else if (row_count < 10000)
        row_block_count = 8;
    else if (row_count < 20000)
        row_block_count = 16;
    else if (row_count < 50000)
        row_block_count = 32;
    else if (row_count < 100000)
        row_block_count = 64;

    return row_block_count;
}

template <typename Float, cov_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_column_block_count(
    std::int64_t column_count) {
    ONEDAL_ASSERT(column_count > 0);

    std::int64_t max_work_group_size = dal::backend::device_max_wg_size(q_);
    return (column_count + max_work_group_size - 1) / max_work_group_size;
}

template <typename Float>
auto compute_covariance(sycl::queue& q,
                        std::int64_t row_count,
                        const pr::ndview<Float, 2>& xtx,
                        const pr::ndarray<Float, 1>& sums,
                        const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_covariance, q);
    ONEDAL_ASSERT(sums.has_data());

    const std::int64_t column_count = xtx.get_dimension(1);

    auto cov =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);

    auto copy_event = copy(q, cov, xtx, { deps });

    auto cov_event = pr::covariance_with_distributed(q, row_count, sums, cov, { copy_event });

    return std::make_tuple(cov, cov_event);
}

template <typename Float>
auto compute_correlation(sycl::queue& q,
                         std::int64_t row_count,
                         const pr::ndview<Float, 2>& xtx,
                         const pr::ndarray<Float, 1>& sums,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, q);
    ONEDAL_ASSERT(sums.has_data());

    const std::int64_t column_count = xtx.get_dimension(1);

    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);

    auto corr =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);

    auto copy_event = copy(q, corr, xtx, { deps });

    auto corr_event =
        pr::correlation_with_distributed(q, row_count, sums, corr, tmp, { copy_event });

    auto smart_event = dal::backend::smart_event{ corr_event }.attach(tmp);
    return std::make_tuple(corr, smart_event);
}

/* single pass kernel for device execution */
template <typename Float, cov_list List, bool DefferedFin>
inline void single_pass_block_processor(const Float* data_ptr,
                                        Float* rsum_ptr,
                                        Float* rsum2cent_ptr,
                                        Float* rmean_ptr,
                                        Float* rvarc_ptr,
                                        std::int64_t row_count,
                                        std::int64_t column_count,
                                        std::int64_t col_block_idx,
                                        std::int64_t column_block_count,
                                        std::int64_t tid,
                                        std::int64_t tnum) {
    const std::int64_t col_offset = col_block_idx * tnum;
    const std::int64_t x = tid + col_offset;

    if (x < column_count) {
        std::int64_t row_block_size = row_count;

        Float sum = Float(0);
        Float mean = Float(0);
        Float sum2cent = Float(0);

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = row * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;

            sum += el;
            mean += delta * inv_n;
            sum2cent += delta * (el - mean);
        }

        rsum_ptr[x] = sum;
        if constexpr (!DefferedFin && check_mask_flag(cov_list::mean, List)) {
            rmean_ptr[x] = mean;
        }
        rsum2cent_ptr[x] = sum2cent;
        Float variance = sum2cent / (row_block_size - Float(1));
        if constexpr (!DefferedFin &&
                      check_mask_flag(cov_list::cov | cov_list::cor | cov_list::mean, List)) {
            rvarc_ptr[x] = variance;
        }
    }
}

/* block processing kernel for device execution */
template <typename Float, cov_list List>
inline void block_processor(const Float* data_ptr,
                            std::int64_t* brc_ptr,
                            Float* bsum_ptr,
                            Float* bsum2cent_ptr,
                            std::int64_t row_count,
                            std::int64_t row_block_idx,
                            std::int64_t row_block_count,
                            std::int64_t column_count,
                            std::int64_t column_block_idx,
                            std::int64_t column_block_count,
                            std::int64_t tid,
                            std::int64_t tnum) {
    const std::int64_t col_offset = column_block_idx * tnum;
    const std::int64_t x = tid + col_offset;

    if (x < column_count) {
        std::int64_t row_block_size = (row_count + row_block_count - 1) / row_block_count;
        const std::int64_t row_offset = row_block_size * row_block_idx;

        if (row_block_size + row_offset > row_count) {
            row_block_size = row_count - row_offset;
        }

        Float sum = Float(0);
        Float mean = Float(0);
        Float sum2cent = Float(0);

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = (row + row_offset) * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;

            sum += el;
            mean += delta * inv_n;
            sum2cent += delta * (el - mean);
        }

        bsum_ptr[x * row_block_count + row_block_idx] = sum;

        bsum2cent_ptr[x * row_block_count + row_block_idx] = sum2cent;
    }
}

/* block processing kernel for device execution */
template <typename Float, cov_list List, bool DefferedFin>
inline void merge_blocks_kernel(sycl::nd_item<1> item,
                                const std::int64_t* brc_ptr,
                                const Float* bsum_ptr,
                                const Float* bsum2cent_ptr,
                                std::int64_t* lrc_ptr,
                                Float* lsum_ptr,
                                Float* lsum2cent_ptr,
                                Float* lmean_ptr,
                                Float* rsum_ptr,
                                Float* rsum2cent_ptr,
                                Float* rmean_ptr,
                                Float* rvarc_ptr,
                                std::int64_t id,
                                std::int64_t group_id,
                                std::int64_t local_size,
                                std::int64_t block_count) {
    Float mrgsum = Float(0);
    Float mrgmean = Float(0);
    Float mrgvectors = Float(0);
    Float mrgsum2cent = Float(0);
    lrc_ptr[id] = 0;

    for (std::int64_t i = id; i < block_count; i += local_size) {
        std::int64_t offset = group_id * block_count + i;

        Float sum = Float(0);
        sum = bsum_ptr[offset];

        std::int64_t rcnt = 1;
        rcnt = brc_ptr[offset];

        Float sum2cent = Float(0);
        sum2cent = bsum2cent_ptr[offset];

        Float mean = sum / static_cast<Float>(rcnt);

        const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
        const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
        const Float delta_scale = mul_n1n2 / sum_n1n2;
        const Float mean_scale = Float(1) / sum_n1n2;
        const Float delta = mean - mrgmean;

        mrgsum += sum;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        mrgvectors = sum_n1n2;

        lrc_ptr[id] += rcnt;

        lsum_ptr[id] = mrgsum;

        lsum2cent_ptr[id] = mrgsum2cent;

        lmean_ptr[id] = mrgmean;
    }

    for (std::int64_t stride = sycl::min(local_size, block_count) / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);

        if (stride > id) {
            std::int64_t offset = id + stride;

            Float sum = Float(0);
            sum = lsum_ptr[offset];

            std::int64_t rcnt = 1;
            rcnt = lrc_ptr[offset];

            Float sum2cent = Float(0);
            sum2cent = lsum2cent_ptr[offset];

            Float mean = Float(0);
            mean = lmean_ptr[offset];

            const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
            const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
            const Float delta_scale = mul_n1n2 / sum_n1n2;
            const Float mean_scale = Float(1) / sum_n1n2;
            const Float delta = mean - mrgmean;

            mrgsum += sum;
            mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
            mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
            mrgvectors = sum_n1n2;

            // item 0 collects all results in private vars
            // but all others need to store it
            if (0 < id) {
                lsum_ptr[id] = mrgsum;

                lrc_ptr[id] += rcnt;

                lsum2cent_ptr[id] = mrgsum2cent;

                lmean_ptr[id] = mrgmean;
            }
        }
    }

    if (0 == id) {
        rsum_ptr[group_id] = mrgsum;

        rsum2cent_ptr[group_id] = mrgsum2cent;

        if constexpr (!DefferedFin) {
            Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));

            rmean_ptr[group_id] = mrgmean;
            if constexpr (check_mask_flag(cov_list::cov | cov_list::cor, List)) {
                rvarc_ptr[group_id] = mrgvariance;
            }
        }
    }
}

template <typename Float, cov_list List>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::merge_blocks(local_buffer_list<Float, List>&& ndbuf,
                                                     std::int64_t column_count,
                                                     std::int64_t block_count,
                                                     const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(merge_blocks, q_);

    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(block_count > 0);

    const bool distr_mode = comm_.get_rank_count() > 1;
    auto ndres = local_result<Float, List>::empty(q_, column_count);

    // ndres asserts
    if (distr_mode) {
        ASSERT_IF(cov_list::mean | cov_list::cor | cov_list::cov,
                  ndres.get_sum().get_count() == column_count);
    }
    else {
        ASSERT_IF(cov_list::mean, ndres.get_sum().get_count() == column_count);
    }
    if (distr_mode) {
        ASSERT_IF(cov_list::cov | cov_list::cor | cov_list::mean,
                  ndres.get_sum2cent().get_count() == column_count);
    }
    else {
        ASSERT_IF(cov_list::mean, ndres.get_sum2cent().get_count() == column_count);
    }
    ASSERT_IF(cov_list::mean, ndres.get_mean().get_count() == column_count);
    ASSERT_IF(cov_list::cov | cov_list::cor, ndres.get_varc().get_count() == column_count);

    // ndbuf asserts
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              ndbuf.get_rc_list().get_count() == block_count * column_count);
    ASSERT_IF(cov_list::mean, ndbuf.get_sum().get_count() == block_count * column_count);

    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              ndbuf.get_sum2cent().get_count() == block_count * column_count);

    const std::int64_t* brc_ptr = ndbuf.get_rc_list().get_data();
    const Float* bsum_ptr = ndbuf.get_sum().get_data();
    const Float* bsum2cent_ptr = ndbuf.get_sum2cent().get_data();

    Float* rsum_ptr = ndres.get_sum().get_mutable_data();

    Float* rsum2cent_ptr = ndres.get_sum2cent().get_mutable_data();

    DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    DECLSET_IF(Float*, rvarc_ptr, cov_list::mean, ndres.get_varc().get_mutable_data())

    std::int64_t local_size = bk::device_max_sg_size(q_);
    auto global_size = de::check_mul_overflow(column_count, local_size);

    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    std::int64_t local_buffer_size = local_size;
    auto last_event = q_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<std::int64_t> lrc_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum2cent_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmean_buf(local_buffer_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t local_size = item.get_local_range()[0];
            const std::int64_t id = item.get_local_id()[0];
            const std::int64_t group_id = item.get_group().get_id(0);

            std::int64_t* lrc_ptr = lrc_buf.get_pointer().get();
            Float* lsum_ptr = lsum_buf.get_pointer().get();
            Float* lsum2cent_ptr = lsum2cent_buf.get_pointer().get();
            Float* lmean_ptr = lmean_buf.get_pointer().get();

            if (distr_mode) {
                merge_blocks_kernel<Float, List, deffered_fin_true>(item,
                                                                    brc_ptr,
                                                                    bsum_ptr,
                                                                    bsum2cent_ptr,
                                                                    lrc_ptr,
                                                                    lsum_ptr,
                                                                    lmean_ptr,
                                                                    lsum2cent_ptr,
                                                                    rsum_ptr,
                                                                    rsum2cent_ptr,
                                                                    rmean_ptr,
                                                                    rvarc_ptr,
                                                                    id,
                                                                    group_id,
                                                                    local_size,
                                                                    block_count);
            }
            else {
                merge_blocks_kernel<Float, List, deffered_fin_false>(item,
                                                                     brc_ptr,
                                                                     bsum_ptr,
                                                                     bsum2cent_ptr,
                                                                     lrc_ptr,
                                                                     lsum_ptr,
                                                                     lmean_ptr,
                                                                     lsum2cent_ptr,
                                                                     rsum_ptr,
                                                                     rsum2cent_ptr,
                                                                     rmean_ptr,
                                                                     rvarc_ptr,
                                                                     id,
                                                                     group_id,
                                                                     local_size,
                                                                     block_count);
            }
        });
    });

    last_event.wait_and_throw();
    return std::make_tuple(std::move(ndres), std::move(last_event));
}

/* merge distributed blocks kernel */
template <typename Float, cov_list List>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::merge_distr_blocks(
    const pr::ndarray<std::int64_t, 1>& com_row_count,
    const pr::ndarray<Float, 1>& com_sum,
    const pr::ndarray<Float, 1>& com_sum2cent,
    local_result<Float, List>&& ndres,
    std::int64_t block_count,
    std::int64_t column_count,
    std::int64_t block_stride, // distance between first elemments of blocks',
    // it can be > column_count for example in case if alignment is ussed
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(block_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(block_stride > 0);

    // ndres asserts
    ASSERT_IF(cov_list::mean | cov_list::cov, ndres.get_sum().get_count() == column_count);
    ASSERT_IF(cov_list::cov | cov_list::cor | cov_list::mean,
              ndres.get_sum2cent().get_count() == column_count);
    ASSERT_IF(cov_list::mean, ndres.get_mean().get_count() == column_count);
    ASSERT_IF(cov_list::cov | cov_list::cor, ndres.get_varc().get_count() == column_count);

    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              com_row_count.get_count() == comm_.get_rank_count());
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              com_sum.get_count() == comm_.get_rank_count() * column_count);
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              com_sum2cent.get_count() == comm_.get_rank_count() * column_count);

    DECLSET_IF(Float*,
               rsum_ptr,
               cov_list::mean | cov_list::cov | cov_list::cor,
               ndres.get_sum().get_mutable_data())
    DECLSET_IF(Float*, rsum2cent_ptr, cov_list::mean, ndres.get_sum2cent().get_mutable_data())
    DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    DECLSET_IF(Float*, rvarc_ptr, cov_list::mean, ndres.get_varc().get_mutable_data())

    const std::int64_t* brc_ptr = com_row_count.get_data();
    const Float* bsum_ptr = com_sum.get_data();
    const Float* bsum2cent_ptr = com_sum2cent.get_data();

    sycl::range<1> range{ de::integral_cast<size_t>(column_count) };

    auto last_event = q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> id) {
            Float mrgsum = Float(0);

            Float mrgvectors = Float(0);
            Float mrgsum2cent = Float(0);
            Float mrgmean = Float(0);

            for (std::int64_t i = 0; i < block_count; ++i) {
                std::int64_t offset = id + i * block_stride;

                Float sum = Float(0);
                sum = bsum_ptr[offset];

                Float rcnt = static_cast<Float>(brc_ptr[i]);
                Float sum2cent = Float(0);
                sum2cent = bsum2cent_ptr[offset];
                Float mean = sum / rcnt;

                Float sum_n1n2 = mrgvectors + rcnt;
                Float mul_n1n2 = mrgvectors * rcnt;
                Float delta_scale = mul_n1n2 / sum_n1n2;
                Float mean_scale = Float(1) / sum_n1n2;
                Float delta = mean - mrgmean;

                mrgsum += sum;

                mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
                mrgmean = (mrgmean * mrgvectors + mean * rcnt) * mean_scale;
                mrgvectors = sum_n1n2;
            }
            rsum_ptr[id] = mrgsum;
            rmean_ptr[id] = mrgmean;
            Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));

            if constexpr (check_mask_flag(cov_list::cov | cov_list::cor | cov_list::mean, List)) {
                rvarc_ptr[id] = mrgvariance;
            }
        });
    });

    // there is an issue in opencl backend with keeping memory dependencies in events.
    last_event.wait_and_throw();

    return std::make_tuple(std::forward<local_result_t>(ndres), last_event);
}

template <typename Float, cov_list List>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_single_pass(const pr::ndarray<Float, 2> data) {
    ONEDAL_PROFILER_TASK(process_single_block, q_);

    ONEDAL_ASSERT(data.has_data());

    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    std::int64_t row_count = data.get_dimension(0);
    std::int64_t column_count = data.get_dimension(1);

    const bool distr_mode = comm_.get_rank_count() > 1;

    auto ndres = local_result<Float, List>::empty(q_, column_count);

    ASSERT_IF(cov_list::mean | cov_list::cor, ndres.get_sum().get_count() == column_count);

    ASSERT_IF(cov_list::mean | cov_list::cor, ndres.get_sum2cent().get_count() == column_count);

    ASSERT_IF(cov_list::mean | cov_list::cor | cov_list::cov,
              ndres.get_mean().get_count() == column_count);

    ASSERT_IF(cov_list::cov | cov_list::cor, ndres.get_varc().get_count() == column_count);

    const auto column_block_count = get_column_block_count(column_count);

    auto data_ptr = data.get_data();

    Float* rsum_ptr = nullptr;
    SET_IF(rsum_ptr,
           cov_list::mean | cov_list::cov | cov_list::cor,
           ndres.get_sum().get_mutable_data())
    Float* rsum2cent_ptr = nullptr;
    SET_IF(rsum2cent_ptr,
           cov_list::mean | cov_list::cov | cov_list::cor,
           ndres.get_sum2cent().get_mutable_data())
    DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    DECLSET_IF(Float*,
               rvarc_ptr,
               cov_list::cov | cov_list::cor,
               ndres.get_varc().get_mutable_data())

    std::int64_t max_work_group_size = bk::device_max_wg_size(q_);
    auto local_size = (max_work_group_size < column_count) ? max_work_group_size : column_count;
    auto global_size = de::check_mul_overflow(column_block_count, local_size);

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    auto last_event = q_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t tid = item.get_local_id()[0];
            const std::int64_t tnum = item.get_local_range()[0];
            const std::int64_t gid = item.get_group().get_id(0);

            const std::int64_t row_block_idx = gid / column_block_count;
            const std::int64_t col_block_idx = gid - row_block_idx * column_block_count;

            if (distr_mode) {
                single_pass_block_processor<Float, List, deffered_fin_true>(data_ptr,
                                                                            rsum_ptr,
                                                                            rsum2cent_ptr,
                                                                            rmean_ptr,
                                                                            rvarc_ptr,
                                                                            row_count,
                                                                            column_count,
                                                                            col_block_idx,
                                                                            column_block_count,
                                                                            tid,
                                                                            tnum);
            }
            else {
                single_pass_block_processor<Float, List, deffered_fin_false>(data_ptr,
                                                                             rsum_ptr,
                                                                             rsum2cent_ptr,
                                                                             rmean_ptr,
                                                                             rvarc_ptr,
                                                                             row_count,
                                                                             column_count,
                                                                             col_block_idx,
                                                                             column_block_count,
                                                                             tid,
                                                                             tnum);
            }
        });
    });

    return std::make_tuple(std::move(ndres), last_event);
}

template <typename Float, cov_list List>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_by_blocks(const pr::ndarray<Float, 2> data,
                                                          std::int64_t row_block_count) {
    ONEDAL_ASSERT(data.has_data());

    std::int64_t row_count = data.get_dimension(0);
    std::int64_t column_count = data.get_dimension(1);

    const auto column_block_count = get_column_block_count(column_count);
    const auto aux_buf_size = de::check_mul_overflow(row_block_count, column_count);

    auto ndbuf = local_buffer_list<Float, List>::empty(q_, aux_buf_size);

    // // ndbuf asserts
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              ndbuf.get_rc_list().get_count() == aux_buf_size);

    ASSERT_IF(cov_list::mean | cov_list::cor | cov_list::cov,
              ndbuf.get_sum().get_count() == aux_buf_size);
    ASSERT_IF(cov_list::cor | cov_list::cov, ndbuf.get_sum2cent().get_count() == aux_buf_size);

    DECLSET_IF(std::int64_t*,
               ablock_rc_ptr,
               cov_list::mean | cov_list::cor | cov_list::cov,
               ndbuf.get_rc_list().get_mutable_data())
    DECLSET_IF(Float*,
               asum_ptr,
               cov_list::mean | cov_list::cor | cov_list::cov,
               ndbuf.get_sum().get_mutable_data())

    DECLSET_IF(Float*,
               asum2cent_ptr,
               cov_list::mean | cov_list::cor | cov_list::cov,
               ndbuf.get_sum2cent().get_mutable_data())

    auto data_ptr = data.get_data();

    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
    auto local_size = (max_work_group_size < column_count) ? max_work_group_size : column_count;
    auto global_size = de::check_mul_overflow(row_block_count * column_block_count, local_size);

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    sycl::event last_event;
    {
        ONEDAL_PROFILER_TASK(process_blocks, q_);
        last_event = q_.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
                const std::int64_t tid = item.get_local_id()[0];
                const std::int64_t tnum = item.get_local_range()[0];
                const std::int64_t gid = item.get_group().get_id(0);

                const std::int64_t row_block_idx = gid / column_block_count;
                const std::int64_t col_block_idx = gid - row_block_idx * column_block_count;

                block_processor<Float, List>(data_ptr,
                                             ablock_rc_ptr,
                                             asum_ptr,
                                             asum2cent_ptr,
                                             row_count,
                                             row_block_idx,
                                             row_block_count,
                                             column_count,
                                             col_block_idx,
                                             column_block_count,
                                             tid,
                                             tnum);
            });
        });
    }

    auto [ndres, merge_event] =
        merge_blocks(std::move(ndbuf), column_count, row_block_count, { last_event });

    return std::make_tuple(std::move(ndres), merge_event);
}

template <typename Float, cov_list List>
std::tuple<local_result<Float, List>, sycl::event> compute_kernel_dense_impl<Float, List>::finalize(
    local_result_t&& ndres,
    std::int64_t row_count,
    std::int64_t column_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(finalize, q_);

    const bool distr_mode = comm_.get_rank_count() > 1;
    // ndres asserts
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              ndres.get_sum().get_count() == column_count);
    ASSERT_IF(cov_list::mean | cov_list::cov | cov_list::cor,
              ndres.get_sum2cent().get_count() == column_count);

    sycl::event last_event;

    if (distr_mode) {
        pr::ndarray<std::int64_t, 1> com_row_count;
        pr::ndarray<Float, 1> com_sum;
        pr::ndarray<Float, 1> com_sum2cent;

        if constexpr (check_mask_flag(cov_list::mean | cov_list::cov | cov_list::cor, List)) {
            auto com_row_count_host =
                pr::ndarray<std::int64_t, 1>::empty({ comm_.get_rank_count() });
            comm_.allgather(row_count, com_row_count_host.flatten()).wait();
            com_row_count = com_row_count_host.to_device(q_);

            de::check_mul_overflow(comm_.get_rank_count(), column_count);
            // sum is required for computing derived statistics, therefore it is suitable to get it by blocks instead of reducing
            com_sum = pr::ndarray<Float, 1>::empty(q_,
                                                   { comm_.get_rank_count() * column_count },
                                                   alloc::device);
            comm_.allgather(ndres.get_sum().flatten(q_, deps), com_sum.flatten(q_)).wait();
        }
        if constexpr (check_mask_flag(cov_list::mean | cov_list::cov | cov_list::cor, List)) {
            com_sum2cent = pr::ndarray<Float, 1>::empty(q_,
                                                        { comm_.get_rank_count() * column_count },
                                                        alloc::device);
            comm_.allgather(ndres.get_sum2cent().flatten(q_, deps), com_sum2cent.flatten(q_))
                .wait();
        }
        if constexpr (check_mask_flag(cov_list::mean | cov_list::cov | cov_list::cor, List)) {
            auto [merge_res, merge_event] = merge_distr_blocks(com_row_count,
                                                               com_sum,
                                                               com_sum2cent,
                                                               std::forward<local_result_t>(ndres),
                                                               comm_.get_rank_count(),
                                                               column_count,
                                                               column_count);
            ndres = merge_res;
            last_event = merge_event;
        }
    }
    else {
        sycl::event::wait_and_throw(deps);
    }

    return std::make_tuple(std::forward<local_result_t>(ndres), std::move(last_event));
}

template <typename Float, cov_list List>
result_t compute_kernel_dense_impl<Float, List>::operator()(const descriptor_t& desc,
                                                            const input_t& input) {
    const auto data = input.get_data();
    std::int64_t row_count = data.get_row_count();
    auto rows_count_global = row_count;
    std::int64_t column_count = data.get_column_count();
    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);
    const auto row_block_count = get_row_block_count(row_count);
    auto [ndres, last_event] = (row_block_count > 1) ? compute_by_blocks(data_nd, row_block_count)
                                                     : compute_single_pass(data_nd);
    last_event.wait_and_throw();
    auto xtx =
        pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, sycl::usm::alloc::device);

    auto gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1), Float(0));
    gemm_event.wait_and_throw();

    comm_.allreduce(xtx.flatten(q_, { last_event }), spmd::reduce_op::sum).wait();

    comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
    std::tie(ndres, last_event) =
        finalize(std::move(ndres), row_count, column_count, { last_event });
    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto [cov, cov_event] =
            compute_covariance(q_, rows_count_global, xtx, ndres.get_sum(), { last_event });
        result.set_cov_matrix(
            (homogen_table::wrap(cov.flatten(q_, { cov_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto [corr, corr_event] =
            compute_correlation(q_, rows_count_global, xtx, ndres.get_sum(), { last_event });

        result.set_cor_matrix(
            (homogen_table::wrap(corr.flatten(q_, { corr_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::means)) {
        result.set_means(
            homogen_table::wrap(ndres.get_mean().flatten(q_, { last_event }), 1, column_count));
    }

    return result;
}

#define INSTANTIATE(LIST)                                  \
    template class compute_kernel_dense_impl<float, LIST>; \
    template class compute_kernel_dense_impl<double, LIST>;

INSTANTIATE(cov_mode_mean);
INSTANTIATE(cov_mode_cov);
INSTANTIATE(cov_mode_cor);
INSTANTIATE(cov_mode_cov_mean);
INSTANTIATE(cov_mode_cov_cor);
INSTANTIATE(cov_mode_cor_mean);
INSTANTIATE(cov_mode_all);

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL
