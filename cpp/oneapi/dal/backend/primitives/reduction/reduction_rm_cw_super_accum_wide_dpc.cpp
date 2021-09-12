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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/super_accumulator.hpp"

namespace oneapi::dal::backend::primitives {

template <bool strided>
struct dimensions_keeper {
    dimensions_keeper(std::int32_t width, std::int32_t height)
            : width_(std::move(width)),
              height_(std::move(height)){};

    const auto& get_stride() const {
        return width_;
    }

    const auto& get_width() const {
        return width_;
    }

    const auto& get_height() const {
        return height_;
    }

    const std::int32_t width_;
    const std::int32_t height_;
};

template <>
struct dimensions_keeper<true> : public dimensions_keeper<false> {
    using base_t = dimensions_keeper<false>;

    dimensions_keeper(std::int32_t width, std::int32_t stride, std::int32_t height)
            : base_t(std::move(width), std::move(height)),
              stride_(std::move(stride)){};

    const auto& get_stride() const {
        return stride_;
    }

    const std::int32_t stride_;
};

template <typename Float, typename UnaryOp, int f, int b, bool s>
class reduction_kernel {
    using super_accums = super_accumulators<Float, false>;
    using dkeeper = dimensions_keeper<s>;

public:
    constexpr static inline float zero = 0.f;
    constexpr static inline int folding = f;
    constexpr static inline int block = b;
    constexpr static inline bool strided = s;

    reduction_kernel(std::int32_t width,
                     std::int32_t stride,
                     std::int32_t height,
                     const Float* data,
                     std::int64_t* bins,
                     const UnaryOp& unary)
            : dk(std::move(width), std::move(stride), std::move(height)),
              data_(std::move(data)),
              bins_(std::move(bins)),
              unary_(std::move(unary)) {}

    reduction_kernel(std::int32_t width,
                     std::int32_t height,
                     const Float* data,
                     std::int64_t* bins,
                     const UnaryOp& unary)
            : dk(std::move(width), std::move(height)),
              data_(std::move(data)),
              bins_(std::move(bins)),
              unary_(std::move(unary)) {}

    void operator()(sycl::nd_item<2> it) const {
        // Acumulators for working in width
        Float accs[folding] = { zero };
        //
        const std::int32_t vid = it.get_global_id(1);
        const std::int32_t hid = it.get_global_id(0);
        const std::int32_t hwg = it.get_global_range(0);
        //
        // Reduction Section
        //
        for (std::int32_t i = 0; i < block; ++i) {
            // Current dataset row
            const std::int64_t rid = vid * block + i;
            for (std::int32_t j = 0; j < folding; ++j) {
                // Current dataset col number
                const auto cid = hid + j * hwg;
                // Check for row and col to be in dataset
                const bool handle = (rid < dk.get_height()) && (cid < dk.get_width());
                // Access to the value in row-major order
                // All arithmetics should work in std::int64_t
                const auto& val = data_[cid + rid * dk.get_stride()];
                accs[j] += handle ? unary_(val) : zero;
            }
        }
        //
        // Super counter Section
        //
        for (std::int32_t j = 0; j < folding; ++j) {
            const auto cid = hid + j * hwg;
            if (cid < dk.get_width()) {
                bins_.add(accs[j], cid);
            }
        }
    }

private:
    const dkeeper dk;
    const Float* const data_;
    const super_accums bins_;
    const UnaryOp unary_;
};

template <typename Float,
          typename UnaryOp,
          int folding,
          int block_size,
          bool strided,
          typename = std::enable_if_t<folding != 0>>
sycl::event reduction_impl(sycl::queue& queue,
                           const Float* data,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           std::int64_t* bins,
                           const UnaryOp& unary = {},
                           const std::vector<sycl::event>& deps = {}) {
    using kernel_t = reduction_kernel<Float, UnaryOp, folding, block_size, strided>;
    constexpr int bl = kernel_t::block;
    const auto n_blocks = height / bl + bool(height % bl);
    const auto wg = std::min<std::int64_t>(device_max_wg_size(queue), width);
    const auto cfolding = width / wg + bool(width % wg);
    if (cfolding == folding) {
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            const auto range = make_multiple_nd_range_2d({ wg, n_blocks }, { wg, 1l });
            if constexpr (strided) {
                h.parallel_for<kernel_t>(range,
                                         kernel_t(dal::detail::integral_cast<std::int32_t>(width),
                                                  dal::detail::integral_cast<std::int32_t>(stride),
                                                  dal::detail::integral_cast<std::int32_t>(height),
                                                  data,
                                                  bins,
                                                  unary));
            }
            else {
                h.parallel_for<kernel_t>(range,
                                         kernel_t(dal::detail::integral_cast<std::int32_t>(width),
                                                  dal::detail::integral_cast<std::int32_t>(height),
                                                  data,
                                                  bins,
                                                  unary));
            }
        });
    }
    if constexpr (folding > 1) {
        return reduction_impl<Float, UnaryOp, folding - 1, block_size, strided>(queue,
                                                                                data,
                                                                                width,
                                                                                stride,
                                                                                height,
                                                                                bins,
                                                                                unary,
                                                                                deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
}

template <typename Float>
class finalize_kernel {
    using super_accums = super_accumulators<Float, false>;

public:
    finalize_kernel(Float* results_, std::int64_t* bins_) : results(results_), all_bins(bins_) {}
    void operator()(sycl::id<1> idx) const {
        results[idx] = all_bins.finalize(idx[0]);
    }

private:
    Float* const results;
    const super_accums all_bins;
};

template <typename Float>
sycl::event finalization(sycl::queue& queue,
                         Float* results,
                         std::int64_t width,
                         std::int64_t* bins,
                         const std::vector<sycl::event>& deps = {}) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(make_range_1d(width), finalize_kernel(results, bins));
    });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::reduction_rm_cw_super_accum_wide(
    sycl::queue& q)
        : q_(q) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    auto reduction_event =
        (stride == width) ? reduction_impl<Float, UnaryOp, max_folding, block_size, false>(q_,
                                                                                           input,
                                                                                           width,
                                                                                           stride,
                                                                                           height,
                                                                                           bins,
                                                                                           unary,
                                                                                           deps)
                          : reduction_impl<Float, UnaryOp, max_folding, block_size, true>(q_,
                                                                                          input,
                                                                                          width,
                                                                                          stride,
                                                                                          height,
                                                                                          bins,
                                                                                          unary,
                                                                                          deps);
    return finalization(q_, output, width, bins, { reduction_event });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, bins, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    using oneapi::dal::backend::operator+;
    using super_accums = super_accumulators<float, false>;
    constexpr auto places_per_feature = super_accums::nbins * super_accums::binsize;
    auto [bins, fill_event] =
        ndarray<std::int64_t, 2>::zeros(this->q_, { width, places_per_feature });
    const auto new_deps = deps + fill_event;
    auto* const bins_ptr = bins.get_mutable_data();
    auto reduction_event =
        this->operator()(input, output, width, height, stride, bins_ptr, binary, unary, new_deps);
    reduction_event.wait_and_throw();
    return reduction_event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_super_accum_wide<F, B, U>;

#define INSTANTIATE_FLOAT(B, U)             \
    INSTANTIATE(float, B<float>, U<float>); \
    INSTANTIATE(double, B<double>, U<double>);

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
