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

#pragma once

#include <cstdint>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/detail_flt.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/detail_dbl.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, bool synchronous = true>
class super_accumulators {
    static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>,
                  "Only float & double type is supported for super accumulation for now");
};

template <bool synchronous>
class super_accumulators<float, synchronous> {
public:
    constexpr static inline int nbins = detail::float32::nbins;
    constexpr static inline int min_buffer_size = nbins;

    super_accumulators(std::int64_t* const bins) : all_bins{ bins } {}

    void add(const float& arg, int idx = 0) const {
        using namespace detail::float32;
        auto* const bins = all_bins + idx * nbins;
        const float_u flt(arg);
        const auto bin = bin_idx(flt.expn_);
        const auto mant = flt.sign_ * new_mant(flt.mant_, flt.expn_);
        //Putting data into corr. bin
        if constexpr (synchronous) {
            bins[bin] += mant;
        }
        else {
#ifdef __SYCL_DEVICE_ONLY__
            atomic_global_add(bins + bin, mant);
#else
            bins[bin] += mant;
#endif
        }
    }

    float finalize(int idx = 0) const {
        using namespace detail::float32;
        constexpr int shift = 23;
        const auto* const bins = all_bins + idx * nbins;
        float acc = 0.f;
        for (int i = 0; i < nbins; ++i) {
            const auto epow = std::uint32_t(i * binratio) << shift;
            acc += (float(bins[i]) * mpowf.floatingpoint * duality32{ epow }.floatingpoint);
        }
        return acc;
    }

private:
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    static inline T atomic_global_add(T* ptr, T operand) {
        using address = cl::sycl::access::address_space;
        return cl::sycl::atomic_fetch_add<T, address::global_space>(
            { cl::sycl::multi_ptr<T, address::global_space>{ ptr } },
            operand);
    }
#endif
    std::int64_t* const all_bins;
};

template <bool synchronous>
class super_accumulators<double, synchronous> {
public:
    constexpr static inline int nbins = detail::float64::nbins;
    constexpr static inline int binsize = detail::float64::binsize;
    constexpr static inline int min_buffer_size = binsize * nbins;

    super_accumulators(std::int64_t* const bins) : all_bins{ bins } {}

    void add(const float& arg, int idx = 0) const {
        using namespace detail::float64;
        const double_u flt{ duality64{ arg } };
        auto* const bins = all_bins + binsize * nbins * idx;
        int128_ptr bin(bins + binsize * bin_idx(flt.expn_));
        const int128_raw mant = new_mant(flt.mant_, flt.expn_);
        bin.template add<int128_raw, !synchronous>(flt.sign_ ? mant : -mant);
    }

    double finalize(int idx = 0) const {
        using namespace detail::float64;
        auto* const bins = all_bins + binsize * nbins * idx;
        constexpr int shift = 52;
        double acc = 0;
        for(int i = 0; i < nbins; ++i) {
            const auto epow = std::uint64_t(i * binratio) << shift;
            const int128_ptr binval(bins + binsize * i);
            acc += (double(binval) * mpowd.floatingpoint *
                                    duality64{ epow }.floatingpoint);
        }
        return acc;
    }

private:
    std::int64_t* const all_bins;
};

} // namespace oneapi::dal::backend::primitives
