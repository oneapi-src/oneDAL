#include <CL/sycl.hpp>
#include "benchmark/benchmark.h"
#include "oneapi/dal/test/engine/fixtures_gbench.hpp"
#include <vector>
#include <iostream>

namespace oneapi::dal::test::engine {
template <typename BM>
class BM_test : public gbench_fixture <BM> {
private:
        std::vector<BM> v;
public:
    /*void SetUp(const ::benchmark::State& st) {
        cl::sycl::default_selector default_device;
        cl::sycl::queue queue(default_device); 
    } 
    void TearDown(const ::benchmark::State& st) {}*/
    void fill(::benchmark::State& st) {
        for (auto _ : st) {
                v.push_back(1);   
        }
    }
    void test_queue () {
        cl::sycl::default_selector default_device;
        cl::sycl::queue q(default_device); 
        int q_size = 1; 
        int value = 5;
        int *data = cl::sycl::malloc_shared<int>(q_size, q);
        q.fill<int>(data, value, q_size).wait_and_throw();
        cl::sycl::free(data, q);
    }
};

BENCHMARK_TEMPLATE_F(BM_test, IntSyclQueueTest, int)(::benchmark::State& state) {
    for (auto _ : state)
        test_queue(); 
};


BENCHMARK_TEMPLATE_F(BM_test, IntVectorTest, int)(::benchmark::State& state) {
        this->fill(state);
};

} // namespace oneapi::dal::test::engine