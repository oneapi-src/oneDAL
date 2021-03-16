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

template <class Float, class BinaryOp, class UnaryOp>
class kernel_reduction_rm_rw_wide;

template <class Float, class BinaryOp, class UnaryOp>
class reduction_rm_rw_wide {
public:
    typedef const Float* inp_t;
    typedef Float* out_t;
    typedef kernel_reduction_rm_rw_wide<Float, BinaryOp, UnaryOp> kernel_t;

public:
    reduction_rm_rw_wide(sycl::queue& q_, const std::int64_t wg_);
    reduction_rm_rw_wide(sycl::queue& q_);
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t stride,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::nd_range<2> get_range(const std::int64_t height) const;
    static kernel_t get_kernel(inp_t input,
                               out_t output,
                               const std::int64_t width,
                               const std::int64_t stride,
                               const BinaryOp binary,
                               const UnaryOp unary);

private:
    sycl::queue& q;

public:
    const std::int64_t wg;
};

template <class Float, class BinaryOp, class UnaryOp>
class kernel_reduction_rm_rw_narrow;

template <class Float, class BinaryOp, class UnaryOp>
class reduction_rm_rw_narrow {
public:
    typedef const Float* inp_t;
    typedef Float* out_t;
    typedef kernel_reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp> kernel_t;

public:
    reduction_rm_rw_narrow(sycl::queue& q_, const std::int64_t wg_);
    reduction_rm_rw_narrow(sycl::queue& q_);
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t stride,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::nd_range<1> get_range(const std::int64_t height) const;
    static kernel_t get_kernel(inp_t input,
                               out_t output,
                               const std::int64_t width,
                               const std::int64_t height,
                               const std::int64_t stride,
                               const BinaryOp binary,
                               const UnaryOp unary);

private:
    sycl::queue& q;

public:
    const std::int64_t wg;
};

template <class Float, class BinaryOp, class UnaryOp>
class reduction_rm_rw {
public:
    typedef const Float* inp_t;
    typedef Float* out_t;
    typedef reduction_rm_rw_narrow<Float, BinaryOp, UnaryOp> narrow_t;
    typedef reduction_rm_rw_wide<Float, BinaryOp, UnaryOp> wide_t;

public:
    reduction_rm_rw(sycl::queue& q_);
    enum reduction_method { wide, narrow };
    reduction_method propose_method(std::int64_t width) const;
    sycl::event operator()(const reduction_method method,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const std::int64_t stride,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(const reduction_method method,
                           inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;
    sycl::event operator()(inp_t input,
                           out_t output,
                           const std::int64_t width,
                           const std::int64_t height,
                           const BinaryOp binary = BinaryOp{},
                           const UnaryOp unary = UnaryOp{},
                           const event_vector& deps = {}) const;

private:
    sycl::queue& q;
};

#endif

} // namespace oneapi::dal::backend::primitives
