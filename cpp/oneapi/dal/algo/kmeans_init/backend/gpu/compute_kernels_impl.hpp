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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/rng/rnd_uniform.hpp"

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

template <typename T>
struct copy_first_observations {};

template <typename Float, typename Method>
struct kmeans_init_kernel {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        using msg = dal::detail::error_messages;
        if constexpr (std::is_same_v<Method, kmeans_init::method::plus_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu());
        }

        if constexpr (std::is_same_v<Method, kmeans_init::method::parallel_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu());
        }
        return sycl::event();
    }
};

template <typename Float>
struct kmeans_init_kernel<Float, kmeans_init::method::dense> {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
        const std::int64_t cluster_count = centroids.get_dimension(0);
        const std::int64_t column_count = centroids.get_dimension(1);
        const auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_mutable_data();
        auto copy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<copy_first_observations<Float>>(
                sycl::range<1>(column_count * cluster_count),
                [=](sycl::id<1> idx) {
                    centroids_ptr[idx] = data_ptr[idx];
                });
        });
        return copy_event;
    }
};

template <typename Float>
struct kmeans_init_kernel<Float, kmeans_init::method::random_dense> {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
        const std::uint64_t row_count =
            dal::detail::integral_cast<std::uint64_t>(data.get_dimension(0));
        const std::uint64_t cluster_count =
            dal::detail::integral_cast<std::uint64_t>(centroids.get_dimension(0));
        const std::int64_t column_count = centroids.get_dimension(1);
        const auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_mutable_data();
        dal::detail::check_mul_overflow(
            cluster_count,
            dal::detail::integral_cast<std::uint64_t>(sizeof(std::int32_t)));

        auto indices = pr::ndarray<std::uint64_t, 1>::empty(queue, cluster_count);
        auto indices_ptr = indices.get_mutable_data();

        pr::rnd_uniform<std::uint64_t> rng;

        std::int64_t k = 0;
        for (std::uint64_t i = 0; i < cluster_count; i++) {
            auto cur_elements = pr::ndarray<std::uint64_t, 1>::wrap(indices_ptr + i, 1);
            rng.generate(queue, cur_elements, i, row_count);
            ONEDAL_ASSERT(indices_ptr[i] >= 0)
            std::uint64_t& value = indices_ptr[i];
            for (std::uint64_t j = i; j > 0; j--) {
                if (value == indices_ptr[j - 1]) {
                    value = j - 1;
                }
            }
            if (value >= row_count)
                continue;
            for (std::int64_t j = 0; j < column_count; j++) {
                centroids_ptr[k * column_count + j] = data_ptr[value * column_count + j];
            }
            k++;
        }
        ONEDAL_ASSERT(k == cluster_count);
        return sycl::event();
    }
};

#endif

} // namespace oneapi::dal::kmeans_init::backend
