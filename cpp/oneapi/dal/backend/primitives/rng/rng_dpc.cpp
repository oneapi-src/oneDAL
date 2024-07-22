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
#include "oneapi/dal/backend/primitives/ndarray.hpp"
namespace oneapi::dal::backend::primitives {

template <typename Type, typename Size>
void rng<Type, Size>::uniform(sycl::queue& queue,
                              Size count,
                              Type* dst,
                              std::uint8_t* state,
                              Type a,
                              Type b,
                              const event_vector& deps) {
    // Implementation of uniform

    // auto d = sycl::device(sycl::cpu_selector_v);
    // sycl::queue cpu_queue(d);
    auto engine = oneapi::mkl::rng::load_state<oneapi::mkl::rng::mrg32k3a>(queue, state);

    oneapi::mkl::rng::uniform<Type> distr(a, b);

    auto event = oneapi::mkl::rng::generate(distr, engine, count, dst, { deps });
    event.wait_and_throw();

    mkl::rng::save_state(engine, state);
}

template <typename Type, typename Size>
void rng<Type, Size>::uniform_mt2203(sycl::queue& queue,
                                     Size count,
                                     Type* dst,
                                     std::int64_t seed,
                                     Type a,
                                     Type b,
                                     const event_vector& deps) {
    // Implementation of uniform
    oneapi::mkl::rng::mt2203 engine(queue, seed);

    oneapi::mkl::rng::uniform<Type> distr(a, b);

    auto event = oneapi::mkl::rng::generate(distr, engine, count, dst, { deps });
    event.wait_and_throw();
}

template <typename Type, typename Size>
void rng<Type, Size>::uniform_without_replacement(sycl::queue& queue,
                                                  Size count,
                                                  Type* dst,
                                                  std::uint8_t* state,
                                                  Type a,
                                                  Type b,
                                                  const event_vector& deps) {
    auto engine = oneapi::mkl::rng::load_state<oneapi::mkl::rng::mrg32k3a>(queue, state);

    oneapi::mkl::rng::uniform<float> distr;
    auto local_buf =
        ndarray<std::int32_t, 1>::empty(queue, { std::int64_t(b) }, sycl::usm::alloc::device);
    auto local_buf_ptr = local_buf.get_mutable_data();

    auto random_buf = ndarray<float, 1>::empty(queue, { count }, sycl::usm::alloc::device);
    auto random_buf_ptr = random_buf.get_mutable_data();

    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>{ std::size_t(b) }, [=](sycl::id<1> idx) {
            local_buf_ptr[idx] = idx;
        });
    });
    fill_event.wait_and_throw();

    auto event = oneapi::mkl::rng::generate(distr, engine, count, random_buf_ptr);
    event.wait_and_throw();

    queue
        .submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{ std::size_t(1) }, [=](sycl::id<1> idx) {
                for (std::int64_t i = 0; i < count; ++i) {
                    auto j = i + (size_t)(random_buf_ptr[i] * (float)(b - i));
                    auto tmp = local_buf_ptr[i];
                    local_buf_ptr[i] = local_buf_ptr[j];
                    local_buf_ptr[j] = tmp;
                }
                for (std::int64_t i = 0; i < count; ++i) {
                    dst[i] = local_buf_ptr[i];
                }
            });
        })
        .wait_and_throw();
    mkl::rng::save_state(engine, state);
}

#define INSTANTIATE(F, Size)                                               \
    template ONEDAL_EXPORT void rng<F, Size>::uniform(sycl::queue& queue,  \
                                                      Size count_,         \
                                                      F* dst,              \
                                                      std::uint8_t* state, \
                                                      F a,                 \
                                                      F b,                 \
                                                      const event_vector& deps);

#define INSTANTIATE_FLOAT(Size) \
    INSTANTIATE(float, Size)    \
    INSTANTIATE(double, Size)   \
    INSTANTIATE(int, Size)

INSTANTIATE_FLOAT(std::int64_t);
INSTANTIATE_FLOAT(std::int32_t);

#define INSTANTIATE_WO_REPLACEMENT(F, Size)                                \
    template ONEDAL_EXPORT void rng<F, Size>::uniform_without_replacement( \
        sycl::queue& queue,                                                \
        Size count_,                                                       \
        F* dst,                                                            \
        std::uint8_t* state,                                               \
        F a,                                                               \
        F b,                                                               \
        const event_vector& deps);

#define INSTANTIATE_WO_REPLACEMENT_FLOAT(Size) \
    INSTANTIATE_WO_REPLACEMENT(float, Size)    \
    INSTANTIATE_WO_REPLACEMENT(double, Size)   \
    INSTANTIATE_WO_REPLACEMENT(int, Size)

INSTANTIATE_WO_REPLACEMENT_FLOAT(std::int64_t);
INSTANTIATE_WO_REPLACEMENT_FLOAT(std::int32_t);

#define INSTANTIATE_WO_REPLACEMENT_MT2203(F, Size)                               \
    template ONEDAL_EXPORT void rng<F, Size>::uniform_mt2203(sycl::queue& queue, \
                                                             Size count_,        \
                                                             F* dst,             \
                                                             std::int64_t state, \
                                                             F a,                \
                                                             F b,                \
                                                             const event_vector& deps);

#define INSTANTIATE_WO_REPLACEMENT_MT2203_FLOAT(Size) \
    INSTANTIATE_WO_REPLACEMENT_MT2203(float, Size)    \
    INSTANTIATE_WO_REPLACEMENT_MT2203(double, Size)   \
    INSTANTIATE_WO_REPLACEMENT_MT2203(int, Size)

INSTANTIATE_WO_REPLACEMENT_MT2203_FLOAT(std::int64_t);
INSTANTIATE_WO_REPLACEMENT_MT2203_FLOAT(std::int32_t);

} // namespace oneapi::dal::backend::primitives
