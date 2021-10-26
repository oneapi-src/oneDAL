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
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
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
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

using comm_t = bk::communicator;
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

    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
    return (column_count + max_work_group_size - 1) / max_work_group_size;
}

template <typename Float, cov_list List>
result_t compute_kernel_dense_impl<Float, List>::get_result(const descriptor_t& desc,
                                                            const local_result<Float, List>& ndres,
                                                            std::int64_t column_count,
                                                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(column_count > 0);
    result_t res;

    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    if (res_op.test(result_options::cor_matrix)) {
        //ONEDAL_ASSERT(ndres.get_min().get_count() == column_count);
        res.set_cor_matrix(homogen_table::wrap(ndres.get_cor_matrix().flatten(q_, deps),
                                               column_count,
                                               column_count));
    }
    if (res_op.test(result_options::cor_matrix)) {
        //ONEDAL_ASSERT(ndres.get_min().get_count() == column_count);
        res.set_cov_matrix(homogen_table::wrap(ndres.get_cov_matrix().flatten(q_, deps),
                                               column_count,
                                               column_count));
    }

    if (res_op.test(result_options::means)) {
        ONEDAL_ASSERT(ndres.get_mean().get_count() == column_count);
        res.set_means(homogen_table::wrap(ndres.get_means().flatten(q_, deps), 1, column_count));
    }
    return res;
}

// template <typename Float>
// auto compute_means(sycl::queue& q,
//                    const pr::ndview<Float, 2>& data,
//                    const dal::backend::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(compute_means, q);
//     ONEDAL_ASSERT(data.has_data());
//     const std::int64_t column_count = data.get_dimension(1);
//     auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto reduce_event =
//         pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
//     auto means_event = pr::means(q, data, sums, means, { reduce_event });

//     return std::make_tuple(means, sums, means_event);
// }

// template <typename Float>
// auto compute_covariance(sycl::queue& q,
//                         const pr::ndview<Float, 2>& data,
//                         const pr::ndview<Float, 1>& sums,
//                         pr::ndview<Float, 1>& means,
//                         const dal::backend::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(compute_covariance, q);
//     ONEDAL_ASSERT(data.has_data());
//     ONEDAL_ASSERT(data.get_dimension(1) == sums.get_dimension(0));
//     ONEDAL_ASSERT(data.get_dimension(1) == means.get_dimension(0));
//     const std::int64_t column_count = data.get_dimension(1);
//     auto cov =
//         pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
//     auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto gemm_event = gemm(q, data.t(), data, cov, Float(1), Float(0), deps);

//     auto cov_event = pr::covariance(q, data, sums, means, cov, vars, tmp, gemm_event);

//     return std::make_tuple(cov, tmp, cov_event);
// }

// template <typename Float>
// auto compute_correlation(sycl::queue& q,
//                          const pr::ndview<Float, 2>& data,
//                          const pr::ndview<Float, 1>& sums,
//                          pr::ndview<Float, 1>& means,
//                          const dal::backend::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(compute_correlation, q);
//     ONEDAL_ASSERT(data.has_data());
//     ONEDAL_ASSERT(data.get_dimension(1) == sums.get_dimension(0));
//     ONEDAL_ASSERT(data.get_dimension(1) == means.get_dimension(0));
//     const std::int64_t column_count = data.get_dimension(1);
//     auto corr =
//         pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
//     auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
//     auto gemm_event = gemm(q, data.t(), data, corr, Float(1), Float(0), deps);
//     auto corr_event = pr::correlation(q, data, sums, means, corr, vars, tmp, gemm_event);

//     auto smart_event = dal::backend::smart_event{ corr_event }.attach(tmp);
//     return std::make_tuple(corr, smart_event);
// }

// template <typename Float>
// auto compute_correlation_with_covariance(sycl::queue& q,
//                                          const pr::ndview<Float, 2>& data,
//                                          const pr::ndview<Float, 2>& cov,
//                                          pr::ndview<Float, 1>& tmp,
//                                          const dal::backend::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(compute_correlation_with_covariance, q);
//     ONEDAL_ASSERT(data.has_data());
//     const std::int64_t column_count = data.get_dimension(1);
//     auto corr =
//         pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
//     auto gemm_event = gemm(q, data.t(), data, corr, Float(1), Float(0), deps);
//     auto corr_event = pr::correlation_with_covariance(q, data, cov, corr, tmp, gemm_event);

//     return std::make_tuple(corr, corr_event);
// }

/* single pass kernel for device execution */
template <typename Float, bool DefferedFin>
inline void single_pass_block_processor(const Float* data_ptr,
                                        //Float* nObservations_,
                                        //Float* crossProduct_,
                                        Float* rsums_,
                                        Float* rmeans,
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

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = row * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;
            sum += el;
            mean += delta * inv_n;
        }

        rsums_[x] = sum;
        rmeans[x] = mean;
    }
}

/* block processing kernel for device execution */
template <typename Float>
inline void block_processor(const Float* data_ptr,
                            std::int64_t* brc_ptr,
                            Float* bsum_ptr,
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

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = (row + row_offset) * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;
            sum += el;
            mean += delta * inv_n;
        }
        bsum_ptr[x * row_block_count + row_block_idx] = sum;
    }
}

/* block processing kernel for device execution */
template <typename Float, bool DefferedFin>
inline void merge_blocks_kernel(sycl::nd_item<1> item,
                                const std::int64_t* brc_ptr,
                                const Float* bsum_ptr,
                                std::int64_t* lrc_ptr,
                                Float* lsum_ptr,
                                Float* lmean_ptr,
                                Float* rsums_,
                                std::int64_t id,
                                std::int64_t group_id,
                                std::int64_t local_size,
                                std::int64_t block_count) {
    Float mrgsum = Float(0);
    Float mrgmean = Float(0);
    Float mrgvectors = Float(0);
    lrc_ptr[id] = 0;

    for (std::int64_t i = id; i < block_count; i += local_size) {
        std::int64_t offset = group_id * block_count + i;

        Float sum = Float(0);
        sum = bsum_ptr[offset];
        std::int64_t rcnt = 1;

        rcnt = brc_ptr[offset];

        Float mean = sum / static_cast<Float>(rcnt);

        const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
        // const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
        // const Float delta_scale = mul_n1n2 / sum_n1n2;
        const Float mean_scale = Float(1) / sum_n1n2;
        // const Float delta = mean - mrgmean;

        mrgsum += sum;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        //mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        mrgvectors = sum_n1n2;

        lrc_ptr[id] += rcnt;

        lsum_ptr[id] = mrgsum;

        lmean_ptr[id] = mrgmean;
    }

    for (std::int64_t stride = sycl::min(local_size, block_count) / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);
        if (stride > id) {
            std::int64_t offset = id + stride;
            Float sum = Float(0);
            Float mean = Float(0);
            sum = lsum_ptr[offset];
            mean = lmean_ptr[offset];
            std::int64_t rcnt = 1;
            rcnt = lrc_ptr[offset];

            const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
            //const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
            //const Float delta_scale = mul_n1n2 / sum_n1n2;
            const Float mean_scale = Float(1) / sum_n1n2;
            //const Float delta = mean - mrgmean;

            //mrgmin = sycl::fmin(min, mrgmin);
            //mrgmax = sycl::fmax(max, mrgmax);
            mrgsum += sum;
            //mrgsum2 += sum2;
            //mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
            mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
            mrgvectors = sum_n1n2;

            // item 0 collects all results in private vars
            // but all others need to store it
            if (0 < id) {
                lsum_ptr[id] = mrgsum;
                lrc_ptr[id] += rcnt;
                lmean_ptr[id] = mrgmean;
            }
        }
    }

    if (0 == id) {
        rsums_[group_id] = mrgsum;
    }

    // if constexpr (!DefferedFin) {
    //     Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));
    //     Float mrgstdev = (Float)sqrt(mrgvariance);
    //     if constexpr (check_mask_flag(cov_list::mean, List)) {
    //         rmean_ptr[group_id] = mrgmean;
    //     }
    //     if constexpr (check_mask_flag(cov_list::sorm, List)) {
    //         rsorm_ptr[group_id] = mrgsum2 / mrgvectors;
    //     }
    //     if constexpr (check_mask_flag(cov_list::varc, List)) {
    //         rvarc_ptr[group_id] = mrgvariance;
    //     }
    //     if constexpr (check_mask_flag(cov_list::stdev, List)) {
    //         rstdev_ptr[group_id] = mrgstdev;
    //     }
    //     if constexpr (check_mask_flag(cov_list::vart, List)) {
    //         rvart_ptr[group_id] = mrgstdev / mrgmean;
    //     }
    // }
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
    auto ndres = local_result<Float, List>::empty(q_, column_count, distr_mode);

    // ndres asserts
    // ASSERT_IF(cov_list::min, ndres.get_min().get_count() == column_count);
    // ASSERT_IF(cov_list::max, ndres.get_max().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum, ndres.get_sum().get_count() == column_count);
    // }
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm, ndres.get_sum2().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::varc | cov_list::stdev | cov_list::vart,
    //               ndres.get_sum2cent().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    // }
    // ASSERT_IF(cov_list::mean, ndres.get_mean().get_count() == column_count);
    // ASSERT_IF(cov_list::sorm, ndres.get_sorm().get_count() == column_count);
    // ASSERT_IF(cov_list::varc, ndres.get_varc().get_count() == column_count);
    // ASSERT_IF(cov_list::stdev, ndres.get_stdev().get_count() == column_count);
    // ASSERT_IF(cov_list::vart, ndres.get_vart().get_count() == column_count);

    // // ndbuf asserts
    // ASSERT_IF(cov_list::mean | sum2cent_based_stat,
    //           ndbuf.get_rc_list().get_count() == block_count * column_count);
    // ASSERT_IF(cov_list::min, ndbuf.get_min().get_count() == block_count * column_count);
    // ASSERT_IF(cov_list::max, ndbuf.get_max().get_count() == block_count * column_count);
    // ASSERT_IF(cov_list::sum | cov_list::mean | sum2cent_based_stat,
    //           ndbuf.get_sum().get_count() == block_count * column_count);
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm,
    //           ndbuf.get_sum2().get_count() == block_count * column_count);
    // ASSERT_IF(sum2cent_based_stat, ndbuf.get_sum2cent().get_count() == block_count * column_count);

    const std::int64_t* brc_ptr = ndbuf.get_rc_list().get_data();
    //const Float* bmin_ptr = ndbuf.get_min().get_data();
    //const Float* bmax_ptr = ndbuf.get_max().get_data();
    const Float* bsum_ptr = ndbuf.get_sums().get_data();
    //const Float* bsum2_ptr = ndbuf.get_sum2().get_data();
    //const Float* bsum2cent_ptr = ndbuf.get_sum2cent().get_data();

    // DECLSET_IF(Float*, rmin_ptr, cov_list::min, ndres.get_min().get_mutable_data())
    // DECLSET_IF(Float*, rmax_ptr, cov_list::max, ndres.get_max().get_mutable_data())
    // DECLSET_IF(Float*,
    //            rsum2_ptr,
    //            cov_list::sum2 | cov_list::sorm,
    //            ndres.get_sum2().get_mutable_data())

    Float* rsum_ptr = nullptr;
    // if (distr_mode) {
    //     SET_IF(rsum_ptr, cov_list::mean | sum2cent_based_stat, ndres.get_sum().get_mutable_data())
    // }
    // else {
    //     SET_IF(rsum_ptr, cov_list::sum, ndres.get_sum().get_mutable_data())
    // }

    // Float* rsum2cent_ptr = nullptr;
    // if (distr_mode) {
    //     SET_IF(rsum2cent_ptr,
    //            cov_list::varc | cov_list::stdev | cov_list::vart,
    //            ndres.get_sum2cent().get_mutable_data())
    // }
    // else {
    //     SET_IF(rsum2cent_ptr, cov_list::sum2cent, ndres.get_sum2cent().get_mutable_data())
    // }

    // DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    // DECLSET_IF(Float*, rsorm_ptr, cov_list::sorm, ndres.get_sorm().get_mutable_data())
    // DECLSET_IF(Float*, rvarc_ptr, cov_list::varc, ndres.get_varc().get_mutable_data())
    // DECLSET_IF(Float*, rstdev_ptr, cov_list::stdev, ndres.get_stdev().get_mutable_data())
    // DECLSET_IF(Float*, rvart_ptr, cov_list::vart, ndres.get_vart().get_mutable_data())

    std::int64_t local_size = bk::device_max_sg_size(q_);
    auto global_size = de::check_mul_overflow(column_count, local_size);

    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    std::int64_t local_buffer_size = local_size;
    auto last_event = q_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<std::int64_t> lrc_buf(local_buffer_size, cgh);
        //local_accessor_rw_t<Float> lmin_buf(local_buffer_size, cgh);
        //local_accessor_rw_t<Float> lmax_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum_buf(local_buffer_size, cgh);
        //local_accessor_rw_t<Float> lsum2_buf(local_buffer_size, cgh);
        //local_accessor_rw_t<Float> lsum2cent_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmean_buf(local_buffer_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t local_size = item.get_local_range()[0];
            const std::int64_t id = item.get_local_id()[0];
            const std::int64_t group_id = item.get_group().get_id(0);

            std::int64_t* lrc_ptr = lrc_buf.get_pointer().get();
            //Float* lmin_ptr = lmin_buf.get_pointer().get();
            //Float* lmax_ptr = lmax_buf.get_pointer().get();
            Float* lsum_ptr = lsum_buf.get_pointer().get();
            //Float* lsum2_ptr = lsum2_buf.get_pointer().get();
            //Float* lsum2cent_ptr = lsum2cent_buf.get_pointer().get();
            Float* lmean_ptr = lmean_buf.get_pointer().get();

            if (distr_mode) {
                merge_blocks_kernel<Float, List, deffered_fin_true>(item,
                                                                    brc_ptr,
                                                                    bsum_ptr,
                                                                    lrc_ptr,
                                                                    //lmin_ptr,
                                                                    //lmax_ptr,
                                                                    lsum_ptr,
                                                                    //lsum2_ptr,
                                                                    //lsum2cent_ptr,
                                                                    lmean_ptr,
                                                                    //rmin_ptr,
                                                                    //rmax_ptr,
                                                                    rsum_ptr,
                                                                    //rsum2_ptr,
                                                                    //rsum2cent_ptr,
                                                                    //rsorm_ptr,
                                                                    //rvarc_ptr,
                                                                    //rstdev_ptr,
                                                                    //rvart_ptr,
                                                                    id,
                                                                    group_id,
                                                                    local_size,
                                                                    block_count);
            }
            else {
                merge_blocks_kernel<Float, List, deffered_fin_false>(item,
                                                                     brc_ptr,
                                                                     bsum_ptr,
                                                                     lrc_ptr,
                                                                     //lmin_ptr,
                                                                     //lmax_ptr,
                                                                     lsum_ptr,
                                                                     //lsum2_ptr,
                                                                     //lsum2cent_ptr,
                                                                     lmean_ptr,
                                                                     //rmin_ptr,
                                                                     //rmax_ptr,
                                                                     rsum_ptr,
                                                                     //rsum2_ptr,
                                                                     //rsum2cent_ptr,
                                                                     //rsorm_ptr,
                                                                     //rvarc_ptr,
                                                                     //rstdev_ptr,
                                                                     //rvart_ptr,
                                                                     id,
                                                                     group_id,
                                                                     local_size,
                                                                     block_count);
            }
        });
    });

    const bool is_opencl_backend =
        !q_.get_device().template get_info<sycl::info::device::opencl_c_version>().empty();
    if (is_opencl_backend) {
        // there is an issue in opencl backend with keeping memory dependencies in events.
        last_event.wait_and_throw();
    }
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
    // ASSERT_IF(cov_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm, ndres.get_sum2().get_count() == column_count);
    // ASSERT_IF(cov_list::varc | cov_list::stdev | cov_list::vart,
    //           ndres.get_sum2cent().get_count() == column_count);
    // ASSERT_IF(cov_list::mean, ndres.get_mean().get_count() == column_count);
    // ASSERT_IF(cov_list::sorm, ndres.get_sorm().get_count() == column_count);
    // ASSERT_IF(cov_list::varc, ndres.get_varc().get_count() == column_count);
    // ASSERT_IF(cov_list::stdev, ndres.get_stdev().get_count() == column_count);
    // ASSERT_IF(cov_list::vart, ndres.get_vart().get_count() == column_count);

    // ASSERT_IF(cov_list::mean | sum2cent_based_stat,
    //           com_row_count.get_count() == comm_.get_rank_count());
    // ASSERT_IF(cov_list::mean | sum2cent_based_stat,
    //           com_sum.get_count() == comm_.get_rank_count() * column_count);
    // ASSERT_IF(sum2cent_based_stat,
    //           com_sum2cent.get_count() == comm_.get_rank_count() * column_count);

    // DECLSET_IF(Float*,
    //            rsum_ptr,
    //            cov_list::mean | sum2cent_based_stat,
    //            ndres.get_sum().get_mutable_data())
    // DECLSET_IF(Float*,
    //            rsum2cent_ptr,
    //            cov_list::varc | cov_list::stdev | cov_list::vart,
    //            ndres.get_sum2cent().get_mutable_data())
    // DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    // DECLSET_IF(Float*, rsorm_ptr, cov_list::sorm, ndres.get_sorm().get_mutable_data())
    // DECLSET_IF(Float*, rvarc_ptr, cov_list::varc, ndres.get_varc().get_mutable_data())
    // DECLSET_IF(Float*, rstdev_ptr, cov_list::stdev, ndres.get_stdev().get_mutable_data())
    // DECLSET_IF(Float*, rvart_ptr, cov_list::vart, ndres.get_vart().get_mutable_data())

    //const Float* rsum2_ptr = ndres.get_sum2().get_data();

    const std::int64_t* brc_ptr = com_row_count.get_data();
    const Float* bsum_ptr = com_sum.get_data();
    //const Float* bsum2cent_ptr = com_sum2cent.get_data();

    const sycl::range<1> range{ de::integral_cast<size_t>(column_count) };

    auto last_event = q_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> id) {
            Float mrgsum = Float(0);

            Float mrgvectors = Float(0);
            Float mrgsum2cent = Float(0);
            Float mrgmean = Float(0);

            for (std::int64_t i = 0; i < block_count; ++i) {
                std::int64_t offset = id + i * block_stride;

                Float sum = Float(0);
                ///if constexpr (check_mask_flag(cov_list::sum | cov_list::mean | sum2cent_based_stat,
                //List)) {
                sum = bsum_ptr[offset];
                //}

                Float rcnt = static_cast<Float>(brc_ptr[i]);
                Float mean = sum / rcnt;

                Float sum_n1n2 = mrgvectors + rcnt;
                Float mul_n1n2 = mrgvectors * rcnt;
                Float delta_scale = mul_n1n2 / sum_n1n2;
                Float mean_scale = Float(1) / sum_n1n2;
                Float delta = mean - mrgmean;

                mrgsum += sum;

                //mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
                mrgmean = (mrgmean * mrgvectors + mean * rcnt) * mean_scale;
                mrgvectors = sum_n1n2;
            }
            Float* rsum_ptr = ndres.get_sums().get_mutable_data();
            //if constexpr (check_mask_flag(cov_list::sum, List)) {
            rsum_ptr[id] = mrgsum;
            //}
            Float* rmean_ptr = ndres.get_mean().get_mutable_data();
            //if constexpr (check_mask_flag(cov_list::mean, List)) {
            rmean_ptr[id] = mrgmean;
            //}
        });
    });

    const bool is_opencl_backend =
        !q_.get_device().template get_info<sycl::info::device::opencl_c_version>().empty();
    if (is_opencl_backend) {
        // there is an issue in opencl backend with keeping memory dependencies in events.
        last_event.wait_and_throw();
    }
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

    auto ndres = local_result<Float, List>::empty(q_, column_count, distr_mode);

    // ASSERT_IF(cov_list::min, ndres.get_min().get_count() == column_count);
    // ASSERT_IF(cov_list::max, ndres.get_max().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum, ndres.get_sum().get_count() == column_count);
    // }
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm, ndres.get_sum2().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::varc | cov_list::stdev | cov_list::vart,
    //               ndres.get_sum2cent().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    // }
    // ASSERT_IF(cov_list::mean, ndres.get_mean().get_count() == column_count);
    // ASSERT_IF(cov_list::sorm, ndres.get_sorm().get_count() == column_count);
    // ASSERT_IF(cov_list::varc, ndres.get_varc().get_count() == column_count);
    // ASSERT_IF(cov_list::stdev, ndres.get_stdev().get_count() == column_count);
    // ASSERT_IF(cov_list::vart, ndres.get_vart().get_count() == column_count);

    const auto column_block_count = get_column_block_count(column_count);

    auto data_ptr = data.get_data();

    // DECLSET_IF(Float*, rmin_ptr, cov_list::min, ndres.get_min().get_mutable_data())
    // DECLSET_IF(Float*, rmax_ptr, cov_list::max, ndres.get_max().get_mutable_data())
    // DECLSET_IF(Float*,
    //            rsum2_ptr,
    //            cov_list::sum2 | cov_list::sorm,
    //            ndres.get_sum2().get_mutable_data())

    Float* rsum_ptr = nullptr;
    if (distr_mode) {
        SET_IF(rsum_ptr, cov_list::mean, ndres.get_sum().get_mutable_data())
    }
    else {
        SET_IF(rsum_ptr, cov_list::mean, ndres.get_sum().get_mutable_data())
    }

    DECLSET_IF(Float*, rmean_ptr, cov_list::mean, ndres.get_mean().get_mutable_data())
    //DECLSET_IF(Float*, rsorm_ptr, cov_list::sorm, ndres.get_sorm().get_mutable_data())
    //DECLSET_IF(Float*, rvarc_ptr, cov_list::varc, ndres.get_varc().get_mutable_data())
    //DECLSET_IF(Float*, rstdev_ptr, cov_list::stdev, ndres.get_stdev().get_mutable_data())
    //DECLSET_IF(Float*, rvart_ptr, cov_list::vart, ndres.get_vart().get_mutable_data())

    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
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
                                                                            rmean_ptr,
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
                                                                             rmean_ptr,
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
    // ASSERT_IF(cov_list::mean | sum2cent_based_stat, ndbuf.get_rc_list().get_count() == aux_buf_size);
    // ASSERT_IF(cov_list::min, ndbuf.get_min().get_count() == aux_buf_size);
    // ASSERT_IF(cov_list::max, ndbuf.get_max().get_count() == aux_buf_size);
    // ASSERT_IF(cov_list::sum | cov_list::mean | sum2cent_based_stat,
    //           ndbuf.get_sum().get_count() == aux_buf_size);
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm, ndbuf.get_sum2().get_count() == aux_buf_size);
    // ASSERT_IF(sum2cent_based_stat, ndbuf.get_sum2cent().get_count() == aux_buf_size);

    DECLSET_IF(std::int64_t*, ablock_rc_ptr, cov_list::mean, ndbuf.get_rc_list().get_mutable_data())
    // DECLSET_IF(Float*, amin_ptr, cov_list::min, ndbuf.get_min().get_mutable_data())
    // DECLSET_IF(Float*, amax_ptr, cov_list::max, ndbuf.get_max().get_mutable_data())
    DECLSET_IF(Float*, asum_ptr, cov_list::mean, ndbuf.get_sum().get_mutable_data())
    // DECLSET_IF(Float*,
    //            asum2_ptr,
    //            cov_list::sum2 | cov_list::sorm,
    //            ndbuf.get_sum2().get_mutable_data())
    // DECLSET_IF(Float*, asum2cent_ptr, sum2cent_based_stat, ndbuf.get_sum2cent().get_mutable_data())

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
    // ASSERT_IF(cov_list::min, ndres.get_min().get_count() == column_count);
    // ASSERT_IF(cov_list::max, ndres.get_max().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum, ndres.get_sum().get_count() == column_count);
    // }
    // ASSERT_IF(cov_list::sum2 | cov_list::sorm, ndres.get_sum2().get_count() == column_count);
    // if (distr_mode) {
    //     ASSERT_IF(cov_list::varc | cov_list::stdev | cov_list::vart,
    //               ndres.get_sum2cent().get_count() == column_count);
    // }
    // else {
    //     ASSERT_IF(cov_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    // }

    sycl::event last_event;

    if (distr_mode) {
        pr::ndarray<std::int64_t, 1> com_row_count;
        pr::ndarray<Float, 1> com_sum;
        pr::ndarray<Float, 1> com_sum2cent;

        if constexpr (check_mask_flag(cov_list::mean, List)) {
            auto com_row_count_host =
                pr::ndarray<std::int64_t, 1>::empty({ comm_.get_rank_count() });
            comm_.allgather(&row_count, 1, com_row_count_host.get_mutable_data(), 1).wait();
            com_row_count = com_row_count_host.to_device(q_);

            de::check_mul_overflow(comm_.get_rank_count(), column_count);
            // sum is required for computing derived statistics, therefore it is suitable to get it by blocks instead of reducing
            com_sum = pr::ndarray<Float, 1>::empty(q_,
                                                   { comm_.get_rank_count() * column_count },
                                                   alloc::device);
            comm_
                .allgather(q_,
                           ndres.get_sum().get_data(),
                           column_count,
                           com_sum.get_mutable_data(),
                           column_count)
                .wait();
        }
        else if constexpr (check_mask_flag(cov_list::mean, List)) {
            comm_
                .allreduce(q_,
                           ndres.get_sum().get_data(),
                           ndres.get_sum().get_mutable_data(),
                           ndres.get_sum().get_count(),
                           de::v1::spmd_reduce_op::sum,
                           deps)
                .wait();
        }

        if constexpr (check_mask_flag(cov_list::mean, List)) {
            auto [merge_res, merge_event] = merge_distr_blocks(com_row_count,
                                                               com_sum,
                                                               com_sum2cent,
                                                               std::forward<local_result_t>(ndres),
                                                               comm_.get_rank_count(),
                                                               column_count,
                                                               column_count);
            ndres = std::move(merge_res);
            last_event = std::move(merge_event);
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
    std::int64_t column_count = data.get_column_count();

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    const auto row_block_count = get_row_block_count(row_count);

    auto [ndres, last_event] = (row_block_count > 1) ? compute_by_blocks(data_nd, row_block_count)
                                                     : compute_single_pass(data_nd);

    std::tie(ndres, last_event) =
        finalize(std::move(ndres), row_count, column_count, { last_event });

    return get_result(desc, std::move(ndres), column_count, { last_event })
        .set_result_options(desc.get_result_options());
}

#define INSTANTIATE(LIST)                            \
    template class compute_kernel_dense_impl<float>; \
    template class compute_kernel_dense_impl<double>;

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL