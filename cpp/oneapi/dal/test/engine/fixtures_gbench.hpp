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

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "benchmark/benchmark.h"

namespace oneapi::dal::test::engine {

template <typename BM>
class gbench_fixture : public benchmark::Fixture, public float_algo_fixture<BM> {
public:
    virtual void run_benchmark(::benchmark::State& st) {}
    virtual void generate(const ::benchmark::State& st) {}
    void SetUp (const ::benchmark::State& state) final {
        this->generate(state);
    }
    void run_and_handle_errors(::benchmark::State& st) {
        try {
                this->run_benchmark(st);
            }
            catch (std::exception const& e) {
                st.SkipWithError(e.what());
                return;
            }
#ifdef ONEDAL_DATA_PARALLEL_MOCK
            catch (oneapi::fpk::lapack::exception const& e) {
                st.SkipWithError(e.what());
                return;
            }
#endif
            catch (...) {
                st.SkipWithError("Undefined exception");
                return;
            }
    }
};

#define DEFINE_TEMPLATE_F(FIXTURE, NAME, TYPE) \
    BENCHMARK_TEMPLATE_DEFINE_F(FIXTURE, NAME, TYPE)(::benchmark::State& st) {\
        this->run_and_handle_errors(st); \
    };

#define BM_TEMPLATE_F(FIXTURE, NAME, TYPE) \
    DEFINE_TEMPLATE_F(FIXTURE, NAME, TYPE);\
    BENCHMARK_REGISTER_F(FIXTURE, NAME)

} // namespace oneapi::dal::test::engine