#include <benchmark/benchmark.h>

#include <iostream>

extern int empty();

static void BM_StringCreation(benchmark::State& state) {
  std::cout << "!!!" << std::endl;
  for (auto _ : state) {
    [[maybe_unused]] volatile int r = empty();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}

BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();