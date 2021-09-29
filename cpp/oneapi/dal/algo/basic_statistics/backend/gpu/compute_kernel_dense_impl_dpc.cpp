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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"

using alloc = sycl::usm::alloc;

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

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

template <typename Float, bs_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_row_block_count(std::int64_t row_count) {
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

template <typename Float, bs_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_column_block_count(
    std::int64_t column_count) {
    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
    return (column_count + max_work_group_size - 1) / max_work_group_size;
}

template <typename Float, bs_list List>
result_t compute_kernel_dense_impl<Float, List>::get_result(const descriptor_t& desc,
                                                            const ndresult<Float, List>& ndres,
                                                            const bk::event_vector& deps) {
    result_t res;

    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    if (res_op.test(result_options::min)) {
        res.set_min(homogen_table::wrap(ndres.get_min().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::max)) {
        res.set_max(homogen_table::wrap(ndres.get_max().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::sum)) {
        res.set_sum(homogen_table::wrap(ndres.get_sum().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::sum_squares)) {
        res.set_sum_squares(
            homogen_table::wrap(ndres.get_sum2().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::sum_squares_centered)) {
        res.set_sum_squares_centered(
            homogen_table::wrap(ndres.get_sum2cent().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::mean)) {
        res.set_mean(homogen_table::wrap(ndres.get_mean().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::second_order_raw_moment)) {
        res.set_second_order_raw_moment(
            homogen_table::wrap(ndres.get_sorm().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::variance)) {
        res.set_variance(homogen_table::wrap(ndres.get_varc().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::standard_deviation)) {
        res.set_standard_deviation(
            homogen_table::wrap(ndres.get_stdev().flatten(q_, deps), 1, column_count_));
    }
    if (res_op.test(result_options::variation)) {
        res.set_variation(
            homogen_table::wrap(ndres.get_vart().flatten(q_, deps), 1, column_count_));
    }
    return res;
}

/* single pass kernel for device execution */
template <typename Float, bs_list List, bool DefferedFin>
inline void single_pass_block_processor(const Float* data_ptr,
                                        Float* rmin_ptr,
                                        Float* rmax_ptr,
                                        Float* rsum_ptr,
                                        Float* rsum2_ptr,
                                        Float* rsum2cent_ptr,
                                        Float* rmean_ptr,
                                        Float* rsorm_ptr,
                                        Float* rvarc_ptr,
                                        Float* rstdev_ptr,
                                        Float* rvart_ptr,
                                        const std::int64_t row_count,
                                        const std::int64_t column_count,
                                        const std::int64_t col_block_idx,
                                        const std::int64_t column_block_count,
                                        const std::int64_t tid,
                                        const std::int64_t tnum) {
    const std::int64_t col_offset = col_block_idx * tnum;
    const std::int64_t x = tid + col_offset;

    if (x < column_count) {
        std::int64_t row_block_size = row_count;

        Float min = data_ptr[x];
        Float max = data_ptr[x];

        Float sum = Float(0);
        Float sum2 = Float(0);
        Float sum2cent = Float(0);
        Float mean = Float(0);

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = row * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;

            min = sycl::fmin(el, min);
            max = sycl::fmax(el, max);
            sum += el;
            sum2 += el * el;
            mean += delta * inv_n;
            sum2cent += delta * (el - mean);
        }

        if constexpr (check_mask_flag(bs_list::min, List)) {
            rmin_ptr[x] = min;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            rmax_ptr[x] = max;
        }
        if constexpr (check_mask_flag(bs_list::sum, List) ||
                      (DefferedFin &&
                       check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                           bs_list::stdev | bs_list::vart,
                                       List))) {
            rsum_ptr[x] = sum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            rsum2_ptr[x] = sum2;
        }
        if constexpr (check_mask_flag(bs_list::sum2cent, List) ||
                      (DefferedFin &&
                       check_mask_flag(bs_list::varc | bs_list::stdev | bs_list::vart, List))) {
            rsum2cent_ptr[x] = sum2cent;
        }

        // common vars calculation
        Float variance = sum2cent / (row_block_size - Float(1));
        Float stdev = (Float)sqrt(variance);

        // output assignment
        if constexpr (!DefferedFin && check_mask_flag(bs_list::mean, List)) {
            rmean_ptr[x] = mean;
        }
        if constexpr (!DefferedFin && check_mask_flag(bs_list::sorm, List)) {
            rsorm_ptr[x] = sum2 / row_block_size;
        }
        if constexpr (!DefferedFin && check_mask_flag(bs_list::varc, List)) {
            rvarc_ptr[x] = variance;
        }
        if constexpr (!DefferedFin && check_mask_flag(bs_list::stdev, List)) {
            rstdev_ptr[x] = stdev;
        }
        if constexpr (!DefferedFin && check_mask_flag(bs_list::vart, List)) {
            rvart_ptr[x] = stdev / mean;
        }
    }
}

/* block processing kernel for device execution */
template <typename Float, bs_list List>
inline void block_processor(const Float* data_ptr,
                            std::int64_t* brc_ptr,
                            Float* bmin_ptr,
                            Float* bmax_ptr,
                            Float* bsum_ptr,
                            Float* bsum2_ptr,
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

        Float min = data_ptr[row_offset * column_count + x];
        Float max = data_ptr[row_offset * column_count + x];
        Float sum = Float(0);
        Float sum2 = Float(0);
        Float sum2cent = Float(0);
        Float mean = Float(0);

        for (std::int64_t row = 0; row < row_block_size; ++row) {
            const std::int64_t y = (row + row_offset) * column_count;
            const Float el = data_ptr[y + x];
            Float inv_n = Float(1) / (row + 1);
            Float delta = el - mean;

            min = sycl::fmin(el, min);
            max = sycl::fmax(el, max);
            sum += el;
            sum2 += el * el;
            mean += delta * inv_n;
            sum2cent += delta * (el - mean);
        }

        if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                          bs_list::stdev | bs_list::vart,
                                      List)) {
            brc_ptr[x * row_block_count + row_block_idx] = row_block_size;
        }
        if constexpr (check_mask_flag(bs_list::min, List)) {
            bmin_ptr[x * row_block_count + row_block_idx] = min;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            bmax_ptr[x * row_block_count + row_block_idx] = max;
        }
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                          bs_list::varc | bs_list::stdev | bs_list::vart,
                                      List)) {
            bsum_ptr[x * row_block_count + row_block_idx] = sum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            bsum2_ptr[x * row_block_count + row_block_idx] = sum2;
        }
        if constexpr (check_mask_flag(
                          bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                          List)) {
            bsum2cent_ptr[x * row_block_count + row_block_idx] = sum2cent;
        }
    }
}

/* block processing kernel for device execution */
template <typename Float, bs_list List, bool DefferedFin>
inline void merge_blocks_kernel(sycl::nd_item<1> item,
                                const std::int64_t* brc_ptr,
                                const Float* bmin_ptr,
                                const Float* bmax_ptr,
                                const Float* bsum_ptr,
                                const Float* bsum2_ptr,
                                const Float* bsum2cent_ptr,
                                std::int64_t* lrc_ptr,
                                Float* lmin_ptr,
                                Float* lmax_ptr,
                                Float* lsum_ptr,
                                Float* lsum2_ptr,
                                Float* lsum2cent_ptr,
                                Float* lmean_ptr,
                                Float* rmin_ptr,
                                Float* rmax_ptr,
                                Float* rsum_ptr,
                                Float* rsum2_ptr,
                                Float* rsum2cent_ptr,
                                Float* rmean_ptr,
                                Float* rsorm_ptr,
                                Float* rvarc_ptr,
                                Float* rstdev_ptr,
                                Float* rvart_ptr,
                                std::int64_t id,
                                std::int64_t group_id,
                                std::int64_t local_size,
                                std::int64_t block_count) {
    Float mrgmin = Float(0);
    if constexpr (check_mask_flag(bs_list::min, List)) {
        mrgmin = bmin_ptr[group_id * block_count + id];
    }
    Float mrgmax = Float(0);
    if constexpr (check_mask_flag(bs_list::max, List)) {
        mrgmax = bmax_ptr[group_id * block_count + id];
    }
    Float mrgsum = Float(0);
    Float mrgsum2 = Float(0);
    Float mrgvectors = Float(0);
    Float mrgsum2cent = Float(0);
    Float mrgmean = Float(0);

    if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                      bs_list::stdev | bs_list::vart,
                                  List)) {
        lrc_ptr[id] = 0;
    }

    for (std::int64_t i = id; i < block_count; i += local_size) {
        std::int64_t offset = group_id * block_count + i;

        Float min = Float(0);
        if constexpr (check_mask_flag(bs_list::min, List)) {
            min = bmin_ptr[offset];
        }
        Float max = Float(0);
        if constexpr (check_mask_flag(bs_list::max, List)) {
            max = bmax_ptr[offset];
        }
        Float sum = Float(0);
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                          bs_list::varc | bs_list::stdev | bs_list::vart,
                                      List)) {
            sum = bsum_ptr[offset];
        }
        Float sum2 = Float(0);
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            sum2 = bsum2_ptr[offset];
        }
        std::int64_t rcnt = 1;
        if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                          bs_list::stdev | bs_list::vart,
                                      List)) {
            rcnt = brc_ptr[offset];
        }
        Float sum2cent = Float(0);
        if constexpr (check_mask_flag(
                          bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                          List)) {
            sum2cent = bsum2cent_ptr[offset];
        }
        Float mean = sum / static_cast<Float>(rcnt);

        Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
        Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
        Float delta_scale = mul_n1n2 / sum_n1n2;
        Float mean_scale = Float(1) / sum_n1n2;
        Float delta = mean - mrgmean;

        mrgmin = sycl::fmin(min, mrgmin);
        mrgmax = sycl::fmax(max, mrgmax);
        mrgsum += sum;
        mrgsum2 += sum2;
        mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        mrgvectors = sum_n1n2;

        if constexpr (check_mask_flag(bs_list::min, List)) {
            lmin_ptr[id] = mrgmin;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            lmax_ptr[id] = mrgmax;
        }
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                          bs_list::varc | bs_list::stdev | bs_list::vart,
                                      List)) {
            lsum_ptr[id] = mrgsum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            lsum2_ptr[id] = mrgsum2;
        }
        if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                          bs_list::stdev | bs_list::vart,
                                      List)) {
            lrc_ptr[id] += rcnt;
        }
        if constexpr (check_mask_flag(
                          bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                          List)) {
            lsum2cent_ptr[id] = mrgsum2cent;
        }
        if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                          bs_list::stdev | bs_list::vart,
                                      List)) {
            lmean_ptr[id] = mrgmean;
        }
    }

    for (std::int64_t stride = sycl::min(local_size, block_count) / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);

        if (stride > id) {
            std::int64_t offset = id + stride;

            Float min = Float(0);
            if constexpr (check_mask_flag(bs_list::min, List)) {
                min = lmin_ptr[offset];
            }
            Float max = Float(0);
            if constexpr (check_mask_flag(bs_list::max, List)) {
                max = lmax_ptr[offset];
            }
            Float sum = Float(0);
            if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                              bs_list::varc | bs_list::stdev | bs_list::vart,
                                          List)) {
                sum = lsum_ptr[offset];
            }
            Float sum2 = Float(0);
            if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
                sum2 = lsum2_ptr[offset];
            }
            std::int64_t rcnt = 1;
            if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                              bs_list::stdev | bs_list::vart,
                                          List)) {
                rcnt = lrc_ptr[offset];
            }
            Float sum2cent = Float(0);
            if constexpr (check_mask_flag(
                              bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                              List)) {
                sum2cent = lsum2cent_ptr[offset];
            }
            Float mean = Float(0);
            if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                              bs_list::stdev | bs_list::vart,
                                          List)) {
                mean = lmean_ptr[offset];
            }

            Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
            Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
            Float delta_scale = mul_n1n2 / sum_n1n2;
            Float mean_scale = Float(1) / sum_n1n2;
            Float delta = mean - mrgmean;

            mrgmin = sycl::fmin(min, mrgmin);
            mrgmax = sycl::fmax(max, mrgmax);
            mrgsum += sum;
            mrgsum2 += sum2;
            mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
            mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
            mrgvectors = sum_n1n2;

            // item 0 collects all results in private vars
            // but all others need to store it
            if (0 < id) {
                if constexpr (check_mask_flag(bs_list::min, List)) {
                    lmin_ptr[id] = mrgmin;
                }
                if constexpr (check_mask_flag(bs_list::max, List)) {
                    lmax_ptr[id] = mrgmax;
                }
                if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                                  bs_list::varc | bs_list::stdev | bs_list::vart,
                                              List)) {
                    lsum_ptr[id] = mrgsum;
                }
                if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
                    lsum2_ptr[id] = mrgsum2;
                }
                if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                                  bs_list::stdev | bs_list::vart,
                                              List)) {
                    lrc_ptr[id] += rcnt;
                }
                if constexpr (check_mask_flag(bs_list::sum2cent | bs_list::varc | bs_list::stdev |
                                                  bs_list::vart,
                                              List)) {
                    lsum2cent_ptr[id] = mrgsum2cent;
                }
                if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                                  bs_list::stdev | bs_list::vart,
                                              List)) {
                    lmean_ptr[id] = mrgmean;
                }
            }
        }
    }

    if (0 == id) {
        if constexpr (check_mask_flag(bs_list::min, List)) {
            rmin_ptr[group_id] = mrgmin;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            rmax_ptr[group_id] = mrgmax;
        }
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                          bs_list::varc | bs_list::stdev | bs_list::vart,
                                      List)) {
            rsum_ptr[group_id] = mrgsum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            rsum2_ptr[group_id] = mrgsum2;
        }
        if constexpr (check_mask_flag(
                          bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                          List)) {
            rsum2cent_ptr[group_id] = mrgsum2cent;
        }

        if constexpr (!DefferedFin) {
            Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));
            Float mrgstdev = (Float)sqrt(mrgvariance);
            if constexpr (check_mask_flag(bs_list::mean, List)) {
                rmean_ptr[group_id] = mrgmean;
            }
            if constexpr (check_mask_flag(bs_list::sorm, List)) {
                rsorm_ptr[group_id] = mrgsum2 / mrgvectors;
            }
            if constexpr (check_mask_flag(bs_list::varc, List)) {
                rvarc_ptr[group_id] = mrgvariance;
            }
            if constexpr (check_mask_flag(bs_list::stdev, List)) {
                rstdev_ptr[group_id] = mrgstdev;
            }
            if constexpr (check_mask_flag(bs_list::vart, List)) {
                rvart_ptr[group_id] = mrgstdev / mrgmean;
            }
        }
    }
}

template <typename Float, bs_list List>
sycl::event compute_kernel_dense_impl<Float, List>::merge_blocks(
    const ndbuffer<Float, List>&& ndbuf,
    ndresult<Float, List>& ndres,
    const std::int64_t block_count,
    const bk::event_vector& deps) {
    const std::int64_t* brc_ptr = ndbuf.get_row_count().get_data();
    const Float* bmin_ptr = ndbuf.get_min().get_data();
    const Float* bmax_ptr = ndbuf.get_max().get_data();
    const Float* bsum_ptr = ndbuf.get_sum().get_data();
    const Float* bsum2_ptr = ndbuf.get_sum2().get_data();
    const Float* bsum2cent_ptr = ndbuf.get_sum2cent().get_data();
    Float* rmin_ptr = ndres.get_min().get_mutable_data();
    Float* rmax_ptr = ndres.get_max().get_mutable_data();
    Float* rsum_ptr = ndres.get_sum().get_mutable_data();
    Float* rsum2_ptr = ndres.get_sum2().get_mutable_data();
    Float* rsum2cent_ptr = ndres.get_sum2cent().get_mutable_data();
    Float* rmean_ptr = ndres.get_mean().get_mutable_data();
    Float* rsorm_ptr = ndres.get_sorm().get_mutable_data();
    Float* rvarc_ptr = ndres.get_varc().get_mutable_data();
    Float* rstdev_ptr = ndres.get_stdev().get_mutable_data();
    Float* rvart_ptr = ndres.get_vart().get_mutable_data();

    std::int64_t local_size = bk::device_max_sg_size(q_);
    auto global_size = de::check_mul_overflow(column_count_, local_size);

    const bool distr_mode = comm_.get_rank_count() > 1;
    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    std::int64_t local_buffer_size = local_size;
    auto last_event = q_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<std::int64_t> lrc_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmin_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmax_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum2_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum2cent_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmean_buf(local_buffer_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t local_size = item.get_local_range()[0];
            const std::int64_t id = item.get_local_id()[0];
            const std::int64_t group_id = item.get_group().get_id(0);

            std::int64_t* lrc_ptr = lrc_buf.get_pointer().get();
            Float* lmin_ptr = lmin_buf.get_pointer().get();
            Float* lmax_ptr = lmax_buf.get_pointer().get();
            Float* lsum_ptr = lsum_buf.get_pointer().get();
            Float* lsum2_ptr = lsum2_buf.get_pointer().get();
            Float* lsum2cent_ptr = lsum2cent_buf.get_pointer().get();
            Float* lmean_ptr = lmean_buf.get_pointer().get();

            if (distr_mode) {
                merge_blocks_kernel<Float, List, deffered_fin_true>(item,
                                                                    brc_ptr,
                                                                    bmin_ptr,
                                                                    bmax_ptr,
                                                                    bsum_ptr,
                                                                    bsum2_ptr,
                                                                    bsum2cent_ptr,
                                                                    lrc_ptr,
                                                                    lmin_ptr,
                                                                    lmax_ptr,
                                                                    lsum_ptr,
                                                                    lsum2_ptr,
                                                                    lsum2cent_ptr,
                                                                    lmean_ptr,
                                                                    rmin_ptr,
                                                                    rmax_ptr,
                                                                    rsum_ptr,
                                                                    rsum2_ptr,
                                                                    rsum2cent_ptr,
                                                                    rmean_ptr,
                                                                    rsorm_ptr,
                                                                    rvarc_ptr,
                                                                    rstdev_ptr,
                                                                    rvart_ptr,
                                                                    id,
                                                                    group_id,
                                                                    local_size,
                                                                    block_count);
            }
            else {
                merge_blocks_kernel<Float, List, deffered_fin_false>(item,
                                                                     brc_ptr,
                                                                     bmin_ptr,
                                                                     bmax_ptr,
                                                                     bsum_ptr,
                                                                     bsum2_ptr,
                                                                     bsum2cent_ptr,
                                                                     lrc_ptr,
                                                                     lmin_ptr,
                                                                     lmax_ptr,
                                                                     lsum_ptr,
                                                                     lsum2_ptr,
                                                                     lsum2cent_ptr,
                                                                     lmean_ptr,
                                                                     rmin_ptr,
                                                                     rmax_ptr,
                                                                     rsum_ptr,
                                                                     rsum2_ptr,
                                                                     rsum2cent_ptr,
                                                                     rmean_ptr,
                                                                     rsorm_ptr,
                                                                     rvarc_ptr,
                                                                     rstdev_ptr,
                                                                     rvart_ptr,
                                                                     id,
                                                                     group_id,
                                                                     local_size,
                                                                     block_count);
            }
        });
    });

    return last_event;
}

/* merge distributed blocks kernel */
template <typename Float, bs_list List>
sycl::event compute_kernel_dense_impl<Float, List>::merge_distr_blocks(
    const pr::ndarray<std::int64_t, 1>& com_row_count,
    const pr::ndarray<Float, 1>& com_sum,
    const pr::ndarray<Float, 1>& com_sum2cent,
    ndresult<Float, List>& ndres,
    std::int64_t block_count,
    std::int64_t column_count,
    std::int64_t block_stride, // distance between first elemments of blocks',
    // it can be > column_count for example in case if alignment is ussed
    const bk::event_vector& deps) {
    Float* rsum_ptr = ndres.get_sum().get_mutable_data();
    Float* rsum2cent_ptr = ndres.get_sum2cent().get_mutable_data();
    Float* rmean_ptr = ndres.get_mean().get_mutable_data();
    Float* rsorm_ptr = ndres.get_sorm().get_mutable_data();
    Float* rvarc_ptr = ndres.get_varc().get_mutable_data();
    Float* rstdev_ptr = ndres.get_stdev().get_mutable_data();
    Float* rvart_ptr = ndres.get_vart().get_mutable_data();

    const Float* rsum2_ptr = ndres.get_sum2().get_mutable_data();

    const std::int64_t* brc_ptr = com_row_count.get_data();
    const Float* bsum_ptr = com_sum.get_data();
    const Float* bsum2cent_ptr = com_sum2cent.get_data();

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
                if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | bs_list::sum2cent |
                                                  bs_list::varc | bs_list::stdev | bs_list::vart,
                                              List)) {
                    sum = bsum_ptr[offset];
                }

                Float rcnt = static_cast<Float>(brc_ptr[i]);
                Float sum2cent = Float(0);
                if constexpr (check_mask_flag(bs_list::sum2cent | bs_list::varc | bs_list::stdev |
                                                  bs_list::vart,
                                              List)) {
                    sum2cent = bsum2cent_ptr[offset];
                }
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

            if constexpr (check_mask_flag(bs_list::sum, List)) {
                rsum_ptr[id] = mrgsum;
            }
            if constexpr (check_mask_flag(bs_list::sum2cent, List)) {
                rsum2cent_ptr[id] = mrgsum2cent;
            }
            if constexpr (check_mask_flag(bs_list::mean, List)) {
                rmean_ptr[id] = mrgmean;
            }
            if constexpr (check_mask_flag(bs_list::sorm, List)) {
                rsorm_ptr[id] = rsum2_ptr[id] / mrgvectors;
            }

            Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));
            Float mrgstdev = sycl::sqrt(mrgvariance);

            if constexpr (check_mask_flag(bs_list::varc, List)) {
                rvarc_ptr[id] = mrgvariance;
            }
            if constexpr (check_mask_flag(bs_list::stdev, List)) {
                rstdev_ptr[id] = mrgstdev;
            }
            if constexpr (check_mask_flag(bs_list::vart, List)) {
                rvart_ptr[id] = mrgstdev / mrgmean;
            }
        });
    });

    return last_event;
}

template <typename Float, bs_list List>
std::tuple<ndresult<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_single_pass(const pr::ndarray<Float, 2> data) {
    ONEDAL_ASSERT(data.get_dimension(0) == row_count_);
    ONEDAL_ASSERT(data.get_dimension(1) == column_count_);

    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    const bool distr_mode = comm_.get_rank_count() > 1;

    auto ndres = ndresult<Float, List>::empty(q_, column_count_, distr_mode);

    std::int64_t row_count = row_count_;
    std::int64_t column_count = column_count_;

    const auto column_block_count = get_column_block_count(column_count_);

    auto data_ptr = data.get_data();

    auto rmin_ptr = ndres.get_min().get_mutable_data();
    auto rmax_ptr = ndres.get_max().get_mutable_data();
    auto rsum_ptr = ndres.get_sum().get_mutable_data();
    auto rsum2_ptr = ndres.get_sum2().get_mutable_data();
    auto rsum2cent_ptr = ndres.get_sum2cent().get_mutable_data();
    auto rmean_ptr = ndres.get_mean().get_mutable_data();
    auto rsorm_ptr = ndres.get_sorm().get_mutable_data();
    auto rvarc_ptr = ndres.get_varc().get_mutable_data();
    auto rstdev_ptr = ndres.get_stdev().get_mutable_data();
    auto rvart_ptr = ndres.get_vart().get_mutable_data();

    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
    auto local_size = (max_work_group_size < column_count_) ? max_work_group_size : column_count_;
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
                                                                            rmin_ptr,
                                                                            rmax_ptr,
                                                                            rsum_ptr,
                                                                            rsum2_ptr,
                                                                            rsum2cent_ptr,
                                                                            rmean_ptr,
                                                                            rsorm_ptr,
                                                                            rvarc_ptr,
                                                                            rstdev_ptr,
                                                                            rvart_ptr,
                                                                            row_count,
                                                                            column_count,
                                                                            col_block_idx,
                                                                            column_block_count,
                                                                            tid,
                                                                            tnum);
            }
            else {
                single_pass_block_processor<Float, List, deffered_fin_false>(data_ptr,
                                                                             rmin_ptr,
                                                                             rmax_ptr,
                                                                             rsum_ptr,
                                                                             rsum2_ptr,
                                                                             rsum2cent_ptr,
                                                                             rmean_ptr,
                                                                             rsorm_ptr,
                                                                             rvarc_ptr,
                                                                             rstdev_ptr,
                                                                             rvart_ptr,
                                                                             row_count,
                                                                             column_count,
                                                                             col_block_idx,
                                                                             column_block_count,
                                                                             tid,
                                                                             tnum);
            }
        });
    });

    return std::make_tuple(ndres, last_event);
}

template <typename Float, bs_list List>
std::tuple<ndresult<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_by_blocks(const pr::ndarray<Float, 2> data,
                                                          std::int64_t row_block_count) {
    ONEDAL_ASSERT(data.get_dimension(0) == row_count_);
    ONEDAL_ASSERT(data.get_dimension(1) == column_count_);

    std::int64_t row_count = row_count_;
    std::int64_t column_count = column_count_;

    const auto column_block_count = get_column_block_count(column_count_);
    const auto aux_buf_size = de::check_mul_overflow(row_block_count, column_count_);

    const bool distr_mode = comm_.get_rank_count() > 1;

    auto ndres = ndresult<Float, List>::empty(q_, column_count_, distr_mode);
    auto ndbuf = ndbuffer<Float, List>::empty(q_, aux_buf_size);

    auto ablock_rc_ptr = ndbuf.get_row_count().get_mutable_data();
    auto amin_ptr = ndbuf.get_min().get_mutable_data();
    auto amax_ptr = ndbuf.get_max().get_mutable_data();
    auto asum_ptr = ndbuf.get_sum().get_mutable_data();
    auto asum2_ptr = ndbuf.get_sum2().get_mutable_data();
    auto asum2cent_ptr = ndbuf.get_sum2cent().get_mutable_data();

    auto data_ptr = data.get_data();

    std::int64_t max_work_group_size =
        q_.get_device().get_info<sycl::info::device::max_work_group_size>();
    auto local_size = (max_work_group_size < column_count_) ? max_work_group_size : column_count_;
    auto global_size = de::check_mul_overflow(row_block_count * column_block_count, local_size);

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    auto last_event = q_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t tid = item.get_local_id()[0];
            const std::int64_t tnum = item.get_local_range()[0];
            const std::int64_t gid = item.get_group().get_id(0);

            const std::int64_t row_block_idx = gid / column_block_count;
            const std::int64_t col_block_idx = gid - row_block_idx * column_block_count;

            block_processor<Float, List>(data_ptr,
                                         ablock_rc_ptr,
                                         amin_ptr,
                                         amax_ptr,
                                         asum_ptr,
                                         asum2_ptr,
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

    last_event = merge_blocks(std::move(ndbuf), ndres, row_block_count, { last_event });

    // wait is required to resolve dependency on internal buffers, smart_event didn't help here by some reason
    last_event.wait_and_throw();
    return std::make_tuple(ndres, last_event);
}

template <typename Float, bs_list List>
std::tuple<ndresult<Float, List>, sycl::event> compute_kernel_dense_impl<Float, List>::finalize(
    ndresult_t&& ndres,
    const bk::event_vector& deps) {
    sycl::event last_event;
    if (comm_.get_rank_count() > 1) {
        if constexpr (check_mask_flag(bs_list::min, List)) {
            comm_
                .allreduce(q_,
                           ndres.get_min().get_data(),
                           ndres.get_min().get_mutable_data(),
                           ndres.get_min().get_count(),
                           de::v1::spmd_reduce_op::min,
                           deps)
                .wait();
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            comm_
                .allreduce(q_,
                           ndres.get_max().get_data(),
                           ndres.get_max().get_mutable_data(),
                           ndres.get_max().get_count(),
                           de::v1::spmd_reduce_op::max,
                           deps)
                .wait();
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            comm_
                .allreduce(q_,
                           ndres.get_sum2().get_data(),
                           ndres.get_sum2().get_mutable_data(),
                           ndres.get_sum2().get_count(),
                           de::v1::spmd_reduce_op::sum,
                           deps)
                .wait();
        }

        pr::ndarray<std::int64_t, 1> com_row_count;
        pr::ndarray<Float, 1> com_sum;
        pr::ndarray<Float, 1> com_sum2cent;

        if constexpr (check_mask_flag(bs_list::mean | bs_list::sum2cent | bs_list::varc |
                                          bs_list::stdev | bs_list::vart,
                                      List)) {
            auto com_row_count_host =
                pr::ndarray<std::int64_t, 1>::empty({ comm_.get_rank_count() });
            comm_.allgather(&row_count_, 1, com_row_count_host.get_mutable_data(), 1).wait();
            com_row_count = com_row_count_host.to_device(q_);

            de::check_mul_overflow(comm_.get_rank_count(), column_count_);
            // sum is required for computing derived statistics, therefore it is suitable to get it by blocks instead of reducing
            com_sum = pr::ndarray<Float, 1>::empty(q_,
                                                   { comm_.get_rank_count() * column_count_ },
                                                   alloc::device);
            comm_
                .allgather(q_,
                           ndres.get_sum().get_data(),
                           column_count_,
                           com_sum.get_mutable_data(),
                           column_count_)
                .wait();
        }
        else if constexpr (check_mask_flag(bs_list::sum, List)) {
            comm_
                .allreduce(q_,
                           ndres.get_sum().get_data(),
                           ndres.get_sum().get_mutable_data(),
                           ndres.get_sum().get_count(),
                           de::v1::spmd_reduce_op::sum,
                           deps)
                .wait();
        }

        if constexpr (check_mask_flag(
                          bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart,
                          List)) {
            com_sum2cent = pr::ndarray<Float, 1>::empty(q_,
                                                        { comm_.get_rank_count() * column_count_ },
                                                        alloc::device);
            comm_
                .allgather(q_,
                           ndres.get_sum2cent().get_data(),
                           column_count_,
                           com_sum2cent.get_mutable_data(),
                           column_count_)
                .wait();
        }

        last_event = merge_distr_blocks(com_row_count,
                                        com_sum,
                                        com_sum2cent,
                                        ndres,
                                        comm_.get_rank_count(),
                                        column_count_,
                                        column_count_);
    }
    else {
        sycl::event::wait_and_throw(deps);
    }

    return std::make_tuple(ndres, last_event);
}

template <typename Float, bs_list List>
result_t compute_kernel_dense_impl<Float, List>::operator()(const descriptor_t& desc,
                                                            const input_t& input) {
    const auto data = input.get_data();

    row_count_ = data.get_row_count();
    column_count_ = data.get_column_count();

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    const auto row_block_count = get_row_block_count(row_count_);

    auto [ndres, last_event] = (row_block_count > 1) ? compute_by_blocks(data_nd, row_block_count)
                                                     : compute_single_pass(data_nd);

    std::tie(ndres, last_event) = finalize(std::move(ndres), { last_event });

    return get_result(desc, std::move(ndres), { last_event })
        .set_result_options(desc.get_result_options());
}

#define INSTANTIATE(LIST)                                  \
    template class compute_kernel_dense_impl<float, LIST>; \
    template class compute_kernel_dense_impl<double, LIST>;

INSTANTIATE(bs_mode_min_max);
INSTANTIATE(bs_mode_mean_variance);
INSTANTIATE(bs_mode_all);

} // namespace oneapi::dal::basic_statistics::backend
