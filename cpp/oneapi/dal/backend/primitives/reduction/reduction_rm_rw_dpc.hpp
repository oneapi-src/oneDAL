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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_rw_wide;

template <typename Float, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw_wide {
public:
    using inp_t = const Float*;
    using out_t = Float*;
    using kernel_t = kernel_reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>;

public:
    reduction_rm_rw_wide(sycl::queue& q, const std::int64_t wg);
    reduction_rm_rw_wide(sycl::queue& q);
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t stride,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::nd_range<2> get_range(const std::int64_t height) const;
    static kernel_t get_kernel(inp_t input,
                               out_t output,
                               const std::int64_t width,
                               const std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary);

private:
    sycl::queue& q_;

public:
    const std::int64_t wg_;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_rw_narrow;

template <typename Float, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw_narrow {
public:
    using inp_t = const Float*;
    using out_t = Float*;
    using kernel_t = kernel_reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>;

public:
    reduction_rm_rw_narrow(sycl::queue& q, const std::int64_t wg);
    reduction_rm_rw_narrow(sycl::queue& q);
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t stride,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::nd_range<1> get_range(const std::int64_t height) const;
    static kernel_t get_kernel(inp_t input,
                               out_t output,
                               const std::int64_t width,
                               const std::int64_t height,
                               const std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary);

private:
    sycl::queue& q_;

private:
    const std::int64_t wg_;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw {
public:
    using inp_t = const Float*;
    using out_t = Float*;
    using narrow_t = reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp>;
    using wide_t = reduction_rm_rw_wide<Float, BinaryOp, UnaryOp>;

public:
    reduction_rm_rw(sycl::queue& q);
    enum reduction_method { wide, narrow };
    reduction_method propose_method(std::int64_t width) const;
    sycl::event operator()(const reduction_method method,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(const reduction_method method,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::queue& q_;
};

#endif

} // namespace oneapi::dal::backend::primitives
