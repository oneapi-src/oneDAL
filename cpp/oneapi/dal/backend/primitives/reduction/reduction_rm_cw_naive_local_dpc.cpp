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

#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_naive_local {
    using acc_t =
        sycl::accessor<Float, 1, sycl::access::mode::read_write, sycl::access::target::local>;

public:
    kernel_reduction_rm_cw_naive_local(acc_t cache,
                                       const Float* input,
                                       Float* output,
                                       std::int64_t height,
                                       std::int32_t lstride,
                                       const BinaryOp& binary,
                                       const UnaryOp& unary)
            : cache_{ cache },
              input_{ input },
              output_{ output },
              unary_{ unary },
              binary_{ binary },
              height_{ height },
              lstride_{ lstride } {}

    void operator()(sycl::nd_item<2> it) const {
        using sycl::ONEAPI::reduce;
        // Common for whole WG
        const auto col_idx = it.get_global_id(0);
        const auto loc_idx = it.get_local_id(1);
        const auto range = it.get_local_range(1);
        const auto lm = cache_.get_count();

        sycl::local_ptr<const Float> local((const Float*)cache_.get_pointer().get());

        Float acc = binary_.init_value;
        // Loop fot the whole WG
        for (std::int64_t j = 0; j < height_; j += lm) {
            const Float* from = input_ + col_idx + lstride_ * j;
            sycl::global_ptr<const Float> global(from);
            const auto count =
                sycl::min(static_cast<std::int32_t>(lm), static_cast<std::int32_t>(height_ - j));
            it.async_work_group_copy<const Float>(local, global, count, lstride_).wait();
            // Exclusive for EU
            for (std::int32_t i = loc_idx; i < count; i += range) {
                acc = binary_.native(acc, unary_(cache_[i]));
            }
        }
        // WG reduction
        auto grp = it.get_group();
        output_[col_idx] = reduce(grp, acc, binary_.native);
    }

private:
    acc_t cache_;
    const Float* const input_;
    Float* const output_;
    const UnaryOp unary_;
    const BinaryOp binary_;
    const std::int64_t height_;
    const std::int32_t lstride_;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::reduction_rm_cw_naive_local(sycl::queue& q,
                                                                                   std::int64_t wg,
                                                                                   std::int64_t lm)
        : q_(q),
          wg_(wg),
          lm_(lm) {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
    ONEDAL_ASSERT(dal::detail::integral_cast<std::int64_t>(2 * lm_ * sizeof(Float)) <=
                  device_local_mem_size(q_));
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::reduction_rm_cw_naive_local(sycl::queue& q)
        : reduction_rm_cw_naive_local(q,
                                      propose_wg_size(q),
                                      device_local_mem_size(q) / sizeof(Float) / 2) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    ONEDAL_ASSERT(0 <= width && width <= stride);
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
    ONEDAL_ASSERT(0 < lm_ && lm_ <= device_local_mem_size(q_));
    auto event = q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(width);
        const auto kernel = get_kernel(h, input, output, lm_, height, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::nd_range<2> reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::get_range(
    std::int64_t width) const {
    return make_multiple_nd_range_2d({ width, wg_ }, { 1, wg_ });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_cw_naive_local<Float, BinaryOp, UnaryOp>::get_kernel(sycl::handler& h,
                                                                  const Float* input,
                                                                  Float* output,
                                                                  std::int64_t lm,
                                                                  std::int64_t height,
                                                                  std::int64_t stride,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary) {
    sycl::accessor<Float, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc{
        sycl::range<1>(lm),
        h
    };
    return kernel_t{ local_acc,
                     input,
                     output,
                     dal::detail::integral_cast<std::int64_t>(height),
                     dal::detail::integral_cast<std::int32_t>(stride),
                     binary,
                     unary };
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_naive_local<F, B, U>;

#define INSTANTIATE_FLOAT(B, U)                \
    INSTANTIATE(double, B<double>, U<double>); \
    INSTANTIATE(float, B<float>, U<float>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
