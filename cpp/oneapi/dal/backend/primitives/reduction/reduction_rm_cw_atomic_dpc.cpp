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
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template<typename T>
constexpr inline bool is_power_of_two(T val) {
    return (val & (val - 1)) == 0;
}

template <typename Float, typename BinaryOp, typename UnaryOp, int f, int b, int r>
class reduction_kernel {
    static_assert(is_power_of_two(r));
public:
    constexpr static inline std::int32_t block = b;
    constexpr static inline std::int32_t folding = f;
    constexpr static inline std::int32_t retbase = r;

    reduction_kernel(const Float* input,
                     Float* output,
                     std::int32_t width,
                     std::int64_t height,
                     std::int32_t lstride,
                     const BinaryOp& binary,
                     const UnaryOp& unary)
            : input_{ input },
              output_{ output },
              unary_{ unary },
              binary_{ binary },
              width_{ width },
              height_{ height },
              lstride_{ lstride } {}

    void operator()(sycl::nd_item<2> it) const {
        Float accs[folding] = { BinaryOp::init_value };
        const std::int32_t vid = it.get_global_id(1);
        const std::int32_t hid = it.get_global_id(0);
        const std::int32_t hwg = it.get_global_range(0);
        for (std::int32_t i = 0; i < block; ++i) {
            const std::int64_t rid = vid * block + i;
            for (std::int32_t j = 0; j < folding; ++j) {
                const auto cid = hid + j * hwg;
                const auto* ptr = input_ + cid + rid * lstride_;
                const bool handle = (rid < height_) && (cid < width_);
                const auto bval = handle ? unary_(*ptr) : BinaryOp::init_value;
                accs[j] = binary_(accs[j], bval);
            }
        }
        //equals       = output_ + (vid % r);
        auto* dst_base = output_ + (vid & (r - 1));
        for(std::int32_t j = 0; j < folding; ++j) {
            const auto cid = hid + j * hwg;
            if(cid < width_) {
                auto* dst_ptr = dst_base + cid * r;
                atomic_binary_op<BinaryOp>(dst_ptr, accs[j]);
            }
        }
    }

private:
    const Float* const input_;
    Float* const output_;
    const UnaryOp unary_;
    const BinaryOp binary_;
    const std::int32_t width_;
    const std::int64_t height_;
    const std::int32_t lstride_;
};

template <typename Float, typename BinaryOp, int r>
class finalize_kernel {
public:
    constexpr static inline std::int32_t retbase = r;
    finalize_kernel(const Float* input,
                    Float* output)
        : input_(input),
          output_(output) {}

    void operator()(sycl::id<1> idx) const {
        const Float* const col = input_ + idx * r;
        Float acc = BinaryOp::init_value;
        for(std::int32_t i = 0; i < r; ++i)
            acc = binary(acc, col[i]);
        *(output_ + idx) = acc;
    }

private:
    const Float* const input_;
    Float* const output_;
    const BinaryOp binary;
};

template <typename Float,
          typename BinaryOp,
          typename UnaryOp,
          int folding,
          int block_size,
          int retbase>
sycl::event reduction_impl(sycl::queue& queue,
                           const Float* data,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           Float* bins,
                           const BinaryOp& binary,
                           const UnaryOp& unary,
                           const event_vector& deps) {
    using kernel_t = reduction_kernel<Float, BinaryOp, UnaryOp, folding, block_size, retbase>;
    constexpr int bl = kernel_t::block;
    const auto n_blocks = height / bl + bool(height % bl);
    const auto wg = std::min<std::int64_t>(device_max_wg_size(queue), width);
    const auto cfolding = width / wg + bool(width % wg);

    if(cfolding == folding) {

        //std::cout << folding << ' ' << n_blocks << ' ' << bl << ' '  << block_size << std::endl;
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            const auto range = make_multiple_nd_range_2d({ wg, n_blocks }, { wg, 1l });
            h.parallel_for<kernel_t>(range,
                                     kernel_t(data,
                                              bins,
                                              dal::detail::integral_cast<std::int32_t>(width),
                                              dal::detail::integral_cast<std::int32_t>(stride),
                                              dal::detail::integral_cast<std::int32_t>(height),
                                              binary,
                                              unary));
        });
    }

    if constexpr (folding > 1) {
        return reduction_impl<Float, BinaryOp, UnaryOp, folding - 1, block_size, retbase>(queue,
                                                                                          data,
                                                                                          width,
                                                                                          stride,
                                                                                          height,
                                                                                          bins,
                                                                                          binary,
                                                                                          unary,
                                                                                          deps);
    }

    ONEDAL_ASSERT(false);
    return sycl::event();
};

template <typename Float, typename BinaryOp, int r>
sycl::event finalization(sycl::queue& queue,
                         const Float* input,
                         Float* output,
                         std::int64_t width,
                         const event_vector& deps) {
    using kernel_t = finalize_kernel<Float, BinaryOp, r>;
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(make_range_1d(width),
                       kernel_t(input, output));
    });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::reduction_rm_cw_atomic(sycl::queue& q) : q_(q) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    Float* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    auto reduction_event = reduction_impl<Float, BinaryOp, UnaryOp, max_folding, block_size, ret_base>(
                                                                                            q_,
                                                                                            input,
                                                                                            width,
                                                                                            stride,
                                                                                            height,
                                                                                            bins,
                                                                                            binary,
                                                                                            unary,
                                                                                            deps);
    return finalization<Float, BinaryOp, ret_base>(q_, bins, output, width, { reduction_event });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    using oneapi::dal::backend::operator+;
    auto [bins, fill_event] =
        ndarray<Float, 2>::full(this->q_, { width, ret_base },
                                binary.init_value, sycl::usm::alloc::device);
    const auto new_deps = deps + fill_event;
    auto* const bins_ptr = bins.get_mutable_data();
    auto reduction_event =
        this->operator()(input, output, width, height, stride, bins_ptr, binary, unary, new_deps);
    reduction_event.wait_and_throw();
    //const auto hbins = bins.to_host(this->q_);
    //std::cout << "bins\n" << hbins << std::endl;
    return reduction_event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_atomic<F, B, U>;

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
