/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <oneapi/mkl.hpp>
#include "oneapi/dal/backend/primitives/rng/rng.hpp"
namespace oneapi::dal::backend::primitives {

void uniform_gen_gpu(sycl::queue& queue,
                     std::int64_t count_,
                     int* dst,
                     std::uint8_t* state,
                     int a,
                     int b) {
    std::int64_t count = static_cast<std::int64_t>(count_);

    auto engine = oneapi::mkl::rng::load_state<oneapi::mkl::rng::mt19937>(queue, state);

    oneapi::mkl::rng::uniform<int> distr(a, b);

    auto event = oneapi::mkl::rng::generate(distr, engine, count, dst, {});
    event.wait_and_throw();

    mkl::rng::save_state(engine, state);
}

} // namespace oneapi::dal::backend::primitives
