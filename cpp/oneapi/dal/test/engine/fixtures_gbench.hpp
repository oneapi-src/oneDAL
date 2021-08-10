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
    };
    void run_and_handle_errors(::benchmark::State& st) {
        try {
                //run_benchmark(st);
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

} // namespace oneapi::dal::test::engine