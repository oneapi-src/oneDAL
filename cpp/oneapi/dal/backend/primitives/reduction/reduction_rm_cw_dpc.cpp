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

#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

inline std::int64_t max_wg(const sycl::queue& q) {
    return q.get_device().template get_info<sycl::info::device::max_work_group_size>();
}

inline std::int64_t local_mem_size(const sycl::queue& q) {
    return q.get_device().template get_info<sycl::info::device::local_mem_size>();
}

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_inplace {
    using inp_t = const Float*;
    using out_t = Float*;

public:
    kernel_reduction_rm_cw_inplace(inp_t const input_,
                                   out_t const output_,
                                   const std::int64_t height_,
                                   const std::int32_t lstride_,
                                   const BinaryOp& binary_,
                                   const UnaryOp& unary_)
            : input{ input_ },
              output{ output_ },
              unary{ unary_ },
              binary{ binary_ },
              height{ height_ },
              lstride{ lstride_ } {}

    void operator()(sycl::nd_item<2> it) const {
        using sycl::ONEAPI::reduce;
        // Common for whole WG
        const auto col_idx = it.get_global_id(0);
        const auto loc_idx = it.get_global_id(1);
        const auto range = it.get_global_range(1);
        // Exclusive for EU
        Float acc = binary.init_value;
        for (std::int64_t i = loc_idx; i < height; i += range) {
            inp_t const inp_row = input + lstride * i;
            acc = binary.native(acc, unary(inp_row[col_idx]));
        }
        // WG reduction
        auto grp = it.get_group();
        output[col_idx] = reduce(grp, acc, binary.native);
    }

private:
    inp_t const input;
    out_t const output;
    const UnaryOp unary;
    const BinaryOp binary;
    const std::int64_t height;
    const std::int32_t lstride;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::reduction_rm_cw_inplace(sycl::queue& q_,
                                                                           const std::int64_t wg_)
        : q(q_),
          wg(wg_) {
    ONEDAL_ASSERT(wg <= max_wg(q));
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::reduction_rm_cw_inplace(sycl::queue& q_)
        : reduction_rm_cw_inplace(q_, max_wg(q_)){};

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    auto event = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(width);
        const auto kernel = get_kernel(input, output, height, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::nd_range<2> reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::get_range(
    const std::int64_t width) const {
    const sycl::range<2> local(1, wg);
    const sycl::range<2> global(width, wg);
    return sycl::nd_range<2>(global, local);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_cw_inplace<Float, BinaryOp, UnaryOp>::get_kernel(const Float* input,
                                                              Float* output,
                                                              const std::int64_t height,
                                                              const std::int64_t stride,
                                                              const BinaryOp& binary,
                                                              const UnaryOp& unary) {
    return kernel_t(input, output, height, stride, binary, unary);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_inplace<F, B, U>;

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

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_inplace_local {
    using acc_t =
        sycl::accessor<Float, 1, sycl::access::mode::read_write, sycl::access::target::local>;
    using inp_t = const Float*;
    using out_t = Float*;

public:
    kernel_reduction_rm_cw_inplace_local(acc_t cache_,
                                         inp_t const input_,
                                         out_t const output_,
                                         const std::int64_t height_,
                                         const std::int32_t lstride_,
                                         const BinaryOp& binary_,
                                         const UnaryOp& unary_)
            : cache{ cache_ },
              input{ input_ },
              output{ output_ },
              unary{ unary_ },
              binary{ binary_ },
              height{ height_ },
              lstride{ lstride_ } {}

    void operator()(sycl::nd_item<2> it) const {
        using sycl::ONEAPI::reduce;
        // Common for whole WG
        const auto col_idx = it.get_global_id(0);
        const auto loc_idx = it.get_global_id(1);
        const auto range = it.get_local_range(1);
        const auto lm = cache.get_count();

        sycl::local_ptr<const Float> local((const Float*)cache.get_pointer().get());

        Float acc = binary.init_value;
        // Loop fot the whole WG
        for (std::int64_t j = 0; j < height; j += lm) {
            inp_t from = input + col_idx + lstride * j;
            sycl::global_ptr<const Float> global(from);
            const auto count = std::min<std::int32_t>(lm, height - j);
            it.async_work_group_copy<const Float>(local, global, count, lstride).wait();
            // Exclusive for EU
            for (std::int32_t i = loc_idx; i < count; i += range) {
                acc = binary.native(acc, unary(cache[i]));
            }
        }
        // WG reduction
        auto grp = it.get_group();
        output[col_idx] = reduce(grp, acc, binary.native);
    }

private:
    acc_t cache;
    inp_t const input;
    out_t const output;
    const UnaryOp unary;
    const BinaryOp binary;
    const std::int64_t height;
    const std::int32_t lstride;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::reduction_rm_cw_inplace_local(
    sycl::queue& q_,
    const std::int64_t wg_,
    const std::int64_t lm_)
        : q(q_),
          wg(wg_),
          lm(lm_) {
    ONEDAL_ASSERT(wg <= max_wg(q));
    ONEDAL_ASSERT(lm * sizeof(Float) <= local_mem_size(q));
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::reduction_rm_cw_inplace_local(
    sycl::queue& q_)
        : reduction_rm_cw_inplace_local(q_, max_wg(q_), local_mem_size(q_) / sizeof(Float) / 2){};

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    auto event = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(width);
        const auto kernel = get_kernel(h, input, output, lm, height, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::nd_range<2> reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::get_range(
    const std::int64_t width) const {
    const sycl::range<2> local(1, wg);
    const sycl::range<2> global(width, wg);
    return sycl::nd_range<2>(global, local);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_cw_inplace_local<Float, BinaryOp, UnaryOp>::get_kernel(sycl::handler& h,
                                                                    const Float* input,
                                                                    Float* output,
                                                                    const std::int64_t lm,
                                                                    const std::int64_t height,
                                                                    const std::int64_t stride,
                                                                    const BinaryOp& binary,
                                                                    const UnaryOp& unary) {
    typedef sycl::accessor<Float, 1, sycl::access::mode::read_write, sycl::access::target::local>
        acc_t;
    acc_t local_acc{ sycl::range<1>(lm), h };
    return kernel_t(local_acc, input, output, height, stride, binary, unary);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_inplace_local<F, B, U>;

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

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw<Float, BinaryOp, UnaryOp>::reduction_rm_cw(sycl::queue& q_) : q{ q_ } {};

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw<Float, BinaryOp, UnaryOp>::reduction_method
reduction_rm_cw<Float, BinaryOp, UnaryOp>::propose_method(std::int64_t width,
                                                          std::int64_t height) const {
    if (height >= max_wg(q) && height > width) {
        return reduction_method::inplace_local;
    }
    return reduction_method::inplace;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(
    const typename reduction_rm_cw<Float, BinaryOp, UnaryOp>::reduction_method method,
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    // TODO: think about `switch` operator
    if (method == reduction_method::inplace) {
        const inplace_t kernel{ q };
        return kernel(input, output, width, height, stride, binary, unary, deps);
    }
    if (method == reduction_method::inplace_local) {
        const inplace_local_t kernel{ q };
        return kernel(input, output, width, height, stride, binary, unary, deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(const Float* input,
                                                                  Float* output,
                                                                  const std::int64_t width,
                                                                  const std::int64_t height,
                                                                  const std::int64_t stride,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    const auto method = propose_method(width, height);
    return this->operator()(method, input, output, width, height, stride, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(
    const typename reduction_rm_cw<Float, BinaryOp, UnaryOp>::reduction_method method,
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(method, input, output, width, height, width, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(const Float* input,
                                                                  Float* output,
                                                                  const std::int64_t width,
                                                                  const std::int64_t height,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    const auto method = propose_method(width, height);
    return this->operator()(method, input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw<F, B, U>;

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

#endif

} // namespace oneapi::dal::backend::primitives
