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
#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common_gbench.hpp"
#include "oneapi/dal/test/engine/fixtures_gbench.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

template <typename Float>
using reduction_types = std::tuple<Float, sum<Float>, square<Float>>;

template <typename Param>
class reduction_rm_cw_gbench : public te::gbench_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    bool is_initialized() const {
        return width_ > 0 && stride_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    void should_be_skipped(::benchmark::State& st) {
        if (this->not_float64_friendly()) {
            st.SkipWithError("Not float64 friendly");
        }
        if (width_ > stride_) {
            st.SkipWithError("width > stride");
        }
    }

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, rm_order>::zeros(this->get_queue(),
                                                    { stride_, height_ },
                                                    sycl::usm::alloc::device);
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(this->get_queue(),
                                                    { size },
                                                    sycl::usm::alloc::device);
    }

    void generate(std::int64_t width, std::int64_t stride, std::int64_t height) {
        this->width_ = width;
        this->stride_ = stride;
        this->height_ = height;
    }

    void generate(const ::benchmark::State& st) final {
        this->generate(st.range(0), st.range(1), st.range(2));
    }

    std::int64_t get_width() const {
        return this->width_;
    }
    std::int64_t get_height() const {
        return this->height_;
    }
    std::int64_t get_stride() const {
        return this->stride_;
    }

    void run_benchmark(::benchmark::State& st) final {
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(this->get_width());

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        this->get_queue().wait_and_throw();

        should_be_skipped(st);

        for (auto _ : st) {
            reduction_rm_cw_naive<float_t, binary_t, unary_t> reducer(this->get_queue());
            reducer(inp_ptr,
                    out_ptr,
                    get_width(),
                    get_height(),
                    get_stride(),
                    binary_t{},
                    unary_t{})
                .wait_and_throw();
        }
    }

private:
    std::int64_t width_;
    std::int64_t stride_;
    std::int64_t height_;
};

#define INSTANTIATE_FLOAT(FPTYPE)                                                                \
    BM_TEMPLATE_F(reduction_rm_cw_gbench, bm_rm_cw_reduction_##FPTYPE, reduction_types<FPTYPE>)  \
        ->ArgsProduct({ { 28, 256, 512, 2000 }, { 28, 256, 512, 2000 }, { 1024, 8192, 32768 } }) \
        ->Unit(benchmark::kMillisecond);

INSTANTIATE_FLOAT(float);
INSTANTIATE_FLOAT(double);

} // namespace oneapi::dal::backend::primitives::test
