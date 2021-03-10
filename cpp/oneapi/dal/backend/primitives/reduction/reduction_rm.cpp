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

#include "oneapi/dal/backend/primitives/reduction/reduction_rm.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

inline auto max_wg(const sycl::queue& q) {
    return q.get_device().template get_info<sycl::info::device::max_work_group_size>();
}

template <class Float, class BinaryOp, class UnaryOp>
class kernel_reduction_rm_rw_wide {
    typedef const Float* inp_t;
    typedef Float* out_t;

public:
    kernel_reduction_rm_rw_wide(inp_t const input_,
                                out_t const output_,
                                const std::int32_t width_,
                                const std::int32_t lstride_,
                                const BinaryOp binary_,
                                const UnaryOp unary_)
            : input{ input_ },
              output{ output_ },
              unary{ unary_ },
              binary{ binary_ },
              width{ width_ },
              lstride{ lstride_ } {}
    void operator()(sycl::nd_item<2> it) const {
        using sycl::ONEAPI::reduce_over_group;
        // Common for whole WG
        const auto row_idx = it.get_global_id(0);
        const auto loc_idx = it.get_global_id(1);
        const auto range = it.get_global_range(1);
        // It should be conmverted to upper type by default
        inp_t const inp_row = input + lstride * row_idx;
        // Exclusive for EU
        Float acc = binary::init_value;
        for (auto i = loc_idx; i < width; i += range) {
            acc = binary(acc, unary(inp_row[i]));
        }
        // WG reduction
        auto grp = it.get_group();
        const auto res = reduce_over_group(grp, acc, binary);
        if (loc_idx == 0) {
            output[row_idx] = res;
        }
    }

private:
    inp_t const input;
    out_t const output;
    const UnaryOp unary;
    const BinaryOp binary;
    const std::int32_t width;
    const std::int32_t lstride;
};

template <class Float, class BinaryOp, class UnaryOp>
reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::reduction_rm_rw_wide(const sycl::queue& q_,
                                                                     const std::int64_t wg_)
        : q(q_),
          wg(wg_) {
    ONEDAL_ASSERT(wg <= max_wg(q));
};

template <class Float, class BinaryOp, class UnaryOp>
reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::reduction_rm_rw_wide(const sycl::queue& q_)
        : reduction_rm_rw_wide(q_, max_wg(q_)){};

template <class Float, class BinaryOp, class UnaryOp>
sycl::event reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const std::int64_t stride,
    const BinaryOp binary,
    const UnaryOp unary,
    const event_vector& deps) const {
    auto event = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(height);
        const auto kernel = get_kernel(input, output, width, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <class Float, class BinaryOp, class UnaryOp>
sycl::event reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const BinaryOp binary,
    const UnaryOp unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

template <class Float, class BinaryOp, class UnaryOp>
sycl::nd_range<2> reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::get_range(
    const std::size_t height) const {
    const sycl::range<2> local(1, wg);
    const sycl::range<2> global(height, wg);
    return sycl::nd_range<2>(local, global);
}

template <class Float, class BinaryOp, class UnaryOp>
typename reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>::get_kernel(const Float* input,
                                                           Float* output,
                                                           const std::int64_t width,
                                                           const std::int64_t stride,
                                                           const BinaryOp binary,
                                                           const UnaryOp unary) {
    return kernel_t(input, output, width, stride, binary, unary);
}

template <class Float, class BinaryOp, class UnaryOp>
class kernel_reduction_rm_rw_narrow {
    kernel_reduction_rm_rw_narrow(inp_t const input_,
                                  out_t const output_,
                                  const std::int32_t width_,
                                  const std::int64_t height_,
                                  const std::int32_t lstride_,
                                  const BinaryOp binary_,
                                  const UnaryOp unary_)
            : input{ input_ },
              output{ output_ },
              unary{ unary_ },
              binary{ binary_ },
              lwidth{ lwidth_ },
              lstride{ lwidth_ } {}
    void operator()(sycl::nd_item<1> it) const {
        // Common for whole WG
        const auto row_idx = it.get_global_id(0);
        if (row_idx < height) {
            // It should be conmverted to upper type by default
            inp_t const inp_row = input + lstride * row_idx;
            // Exclusive for EU
            Float acc = binary::init_value;
            for (auto i = 0; i < width; ++i) {
                acc = binary(acc, unary(inp_row[i]));
            }
            output[row_idx] = res;
        }
    }

private:
    inp_t const input;
    out_t const output;
    const UnaryOp unary;
    const BinaryOp binary;
    const std::int32_t width;
    const std::int64_t height;
    const std::int32_t lstride;
};

template <class Float, class BinaryOp, class UnaryOp>
reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::reduction_rm_rw_narrow(const sycl::queue& q_,
                                                                         const std::int64_t wg_)
        : q(q_),
          wg(wg_) {
    ONEDAL_ASSERT(wg <= max_wg(q));
};

template <class Float, class BinaryOp, class UnaryOp>
reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::reduction_rm_rw_narrow(const sycl::queue& q_)
        : reduction_rm_rw_narrow(q_, max_wg(q_)){};

template <class Float, class BinaryOp, class UnaryOp>
sycl::event reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const std::int64_t stride,
    const BinaryOp binary,
    const UnaryOp unary,
    const event_vector& deps) const {
    auto event = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(height);
        const auto kernel = get_kernel(input, output, width, height, stride, binary, unary);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <class Float, class BinaryOp, class UnaryOp>
sycl::event reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    const std::int64_t width,
    const std::int64_t height,
    const BinaryOp binary,
    const UnaryOp unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

template <class Float, class BinaryOp, class UnaryOp>
sycl::nd_range<2> reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::get_range(
    const std::size_t height) const {
    const sycl::range<1> local(wg);
    const auto nblocks = (height / wg) + bool(height % wg);
    const auto eheight = wg * nblocks;
    const sycl::range<1> global(eheight);
    return sycl::nd_range<1>(local, global);
}

template <class Float, class BinaryOp, class UnaryOp>
typename reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>::get_kernel(inp_t input,
                                                             out_t output,
                                                             const std::int64_t width,
                                                             const std::int64_t height,
                                                             const std::int64_t stride,
                                                             const BinaryOp binary,
                                                             const UnaryOp unary) const {
    return kernel_t(input, output, width, height, stride, binary, unary);
}

template <class Float, class BinaryOp, class UnaryOp>
reduction_rm_rw<Float, BinaryOp, UnaryOp>::reduction_rm_rw(const sycl::queue& q_) : q{ q_ } {};

template <class Float, class BinaryOp, class UnaryOp>
typename reduction_rm_rw<Float, BinaryOp, UnaryOp> reduction_rm_rw<Float, BinaryOp, UnaryOp>::propose_scale(std::int64_t width) const {
        const auto max_wg_size = max_wg(q);
        if (width < max_wg_size) {
            return task_scale::narrow;
        }
        else {
            return task_scale::wide;
        }
        return task_scale::wide;
    }
/*    sycl::event operator()(const task_scale scale,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const {
        // TODO: think about `switch` operator
        if (scale == task_scale::narrow) {
            const narrow_t kernel{ q };
            return kernel(input, output, width, height, stride, binary, unary, deps);
        }
        if (scale == task_scale::wide) {
            const wide_t kernel{ q };
            return kernel(input, output, width, height, stride, binary, unary, deps);
        }
        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
        });
    }
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const {
        const auto scale = propose_scale(width);
        return this->operator()(scale, input, output, width, height, stride, binary, unary, deps);
    }
    sycl::event operator()(const task_scale scale,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const {
        return this->operator()(scale, input, output, width, height, width, binary, unary, deps);
    }
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const {
        return this->operator()(input, output, width, height, width, binary, unary, deps);
    }

private:
    const sycl::queue& q;
};*/

#endif

} // namespace oneapi::dal::backend::primitives
