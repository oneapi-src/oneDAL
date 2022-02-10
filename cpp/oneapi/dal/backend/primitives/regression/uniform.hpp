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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename ResponseType = float>
class uniform_regression {
public:
    virtual sycl::event operator()(const ndview<ResponseType, 2>& responses,
                                   ndview<ResponseType, 1>& results,
                                   const event_vector& deps = {}) = 0;
    virtual ~uniform_regression();

protected:
    uniform_regression(sycl::queue& q);
    sycl::queue& get_queue() const;

private:
    sycl::queue& queue_;
};

template <typename ResponseType = float>
class naive_uniform_regression : public uniform_regression<ResponseType> {
    using base_t = uniform_regression<ResponseType>;

public:
    naive_uniform_regression(sycl::queue& queue);
    sycl::event operator()(const ndview<ResponseType, 2>& responses,
                           ndview<ResponseType, 1>& results,
                           const event_vector& deps = {}) final;

private:
    sycl::event select_winner(ndview<ResponseType, 1>& results, const event_vector& deps) const;
};

template <typename ResponseType = float>
std::unique_ptr<uniform_regression<ResponseType>> make_uniform_regression(sycl::queue& queue,
                                                                          std::int64_t max_block,
                                                                          std::int64_t k_response);

#endif

} // namespace oneapi::dal::backend::primitives
