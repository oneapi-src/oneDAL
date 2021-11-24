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

template<typename T, int size = 8 * sizeof(T)>
constexpr inline int bit_count(T val) {
    constexpr T mask = T{ 0b1 };
    return static_cast<int>((T >> size) & mask) + bit_count<T, size - 1>()
}

template<typename T>
constexpr inline bool is_power_of_two(T val) {
    constexpr T mask{ 0b1 };
    return (val & mask)
}

template<typename Float, typename BinaryOp>


template <typename Float, typename BinaryOp, typename UnaryOp, int b, int f, int r>
class kernel_reduction_rm_cw_atomic {

public:
    constexpr static inline std::int32_t block = b;
    constexpr static inline std::int32_t folding = f;
    constexpr static inline std::int32_t retbase = r;
    constexpr static inline Float init = 0.0;

    kernel_reduction_rm_cw_atomic(const Float* input,
                                 Float* output,
                                 std::int64_t width,
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
        Float accs[folding] = { binary_.init_value };
        const std::int32_t vid = it.get_global_id(1);
        const std::int32_t hid = it.get_global_id(0);
        const std::int32_t hwg = it.get_global_range(0);
        for (std::int32_t i = 0; i < block; ++i) {
            const std::int64_t rid = vid * block + i;
            for (std::int32_t j = 0; j < folding; ++j) {
                const auto cid = hid + j * hwg;
                const auto& val = data_[cid + rid * lstride_];
                const bool handle = (rid < height_) && (cid < width_);
                const auto bval = handle ? unary_(val) : binary_.init_value;
                accs[j] = binary_(accs[j], bval);
            }
        }
        const auto rid = vid % retbase;
        for(std::int32_t j = 0; j < folding; ++j) {

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

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::reduction_rm_cw_atomic(sycl::queue& q,
                                                                       std::int64_t wg)
        : q_(q),
          wg_(wg) {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::reduction_rm_cw_atomic(sycl::queue& q)
        : reduction_rm_cw_atomic(q, propose_wg_size(q)) {}

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
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
    ONEDAL_ASSERT(0 <= width && width <= stride);
    auto event = q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(width);
        const auto kernel = get_kernel(input, output, height, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
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

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::nd_range<2> reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::get_range(
    std::int64_t width) const {
    return make_multiple_nd_range_2d({ width, wg_ }, { 1, wg_ });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_cw_atomic<Float, BinaryOp, UnaryOp>::get_kernel(const Float* input,
                                                            Float* output,
                                                            std::int64_t height,
                                                            std::int64_t stride,
                                                            const BinaryOp& binary,
                                                            const UnaryOp& unary) {
    return kernel_t{ input,
                     output,
                     dal::detail::integral_cast<std::int64_t>(height),
                     dal::detail::integral_cast<std::int32_t>(stride),
                     binary,
                     unary };
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
