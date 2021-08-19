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
#include <type_traits>

#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/test/engine/common_gbench.hpp"
#include "oneapi/dal/test/engine/fixtures_gbench.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class gemm_gbench : public te::gbench_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr ndorder ao = std::tuple_element_t<1, Param>::value;
    static constexpr ndorder bo = std::tuple_element_t<2, Param>::value;
    static constexpr ndorder co = std::tuple_element_t<3, Param>::value;

    void if_should_be_skipped(::benchmark::State& st) {
        if (this->not_float64_friendly()){
            st.SkipWithError("Not float64 friendly");
        }
    }

    void generate(std::int64_t m, std::int64_t n, std::int64_t k) { 
	    this->m_ = m;
        this->n_ = n;
        this->k_ = k;
    }
    
    void generate(const ::benchmark::State& st) final {
       this->generate(st.range(0), st.range(1), st.range(2));
    }

    auto A() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { m_, k_ });
    }

    auto B() {
        check_if_initialized();
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { k_, n_ });
    }

    auto C() {
        check_if_initialized();
        return ndarray<float_t, 2, co>::empty(this->get_queue(), { m_, n_ });
    }

    void run_benchmark(::benchmark::State& st) final {
        auto c = C();
        auto [a, a_e] = A();
        auto [b, b_e] = B();

        this->get_queue().wait_and_throw();

        if_should_be_skipped(st); 

        for (auto _ : st) {
            gemm(this->get_queue(), a, b, c, { a_e, b_e }).wait_and_throw();
        }
    }

    bool is_initialized() const {
        return m_ > 0 && n_ > 0 && k_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "gemm test is not initialized" };
        }
    }

private:
    std::int64_t m_;
    std::int64_t n_;
    std::int64_t k_;
};

template<typename Float>
using gemm_types =  std::tuple<Float, c_order, c_order, c_order>;

#define INSTANTIATE_FLOAT(FPTYPE)\
BM_TEMPLATE_F(gemm_gbench, gemm_##FPTYPE, gemm_types<FPTYPE> )->ArgsProduct({{4, 5, 300, 400},\
                                                                {5, 6, 400, 500}, {6, 7, 500, 600}})->Unit(benchmark::kMillisecond);

INSTANTIATE_FLOAT(float);
INSTANTIATE_FLOAT(double);                                                

} // namespace oneapi::dal::backend::primitives::test
