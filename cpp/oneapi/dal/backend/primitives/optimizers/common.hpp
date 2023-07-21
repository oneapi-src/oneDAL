/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event l1_norm(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps = {});

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps = {});

template <typename Float>
class BaseMatrixOperator {
public:
    virtual ~BaseMatrixOperator() {}
    virtual sycl::event operator()(const ndview<Float, 1>& vec,
                                   ndview<Float, 1>& out,
                                   const event_vector& deps = {}) = 0;
};

template <typename Float>
class LinearMatrixOperator : public BaseMatrixOperator<Float> {
public:
    LinearMatrixOperator(sycl::queue& q, const ndview<Float, 2>& A);
    ~LinearMatrixOperator() {}
    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps) override {
        ONEDAL_ASSERT(A_.get_dimension(1) == vec.get_dimension(0));
        ONEDAL_ASSERT(out.get_dimension(0) == vec.get_dimension(0));
        sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);
        return gemv(q_, A_, vec, out, Float(1), Float(0), { fill_out_event });
    }

private:
    sycl::queue q_;
    const ndview<Float, 2> A_;
};

template <typename Float>
class BaseFunction {
public:
    virtual ~BaseFunction() {}
    virtual Float get_value() = 0;
    virtual ndview<Float, 1>& get_gradient() = 0;
    virtual BaseMatrixOperator<Float>& get_hessian_product() = 0;
    virtual sycl::event update_x(const ndview<Float, 1>& x,
                                 bool needHessp = false,
                                 const event_vector& deps = {}) = 0;
};

// f(x) = 1/2 x^t A x - b^t x
// df / dx = Ax - b
// df / d^2x = A
template <typename Float>
class QuadraticFunction : public BaseFunction<Float> {
public:
    QuadraticFunction(sycl::queue& q, const ndview<Float, 2>& A, const ndview<Float, 1>& b)
            : q_(q),
              n_(A.get_dimension(0)),
              A_(A),
              b_(b),
              hessp_(q, A) {
        ONEDAL_ASSERT(A.get_dimension(1) == n_);
        ONEDAL_ASSERT(b.get_dimension(0) == n_);
        gradient_ = ndarray<Float, 1>::empty(q, { n_ }, sycl::usm::alloc::device);
        tmp_ = ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device);
    }
    ~QuadraticFunction() {}
    Float get_value() override {
        return value_;
    }

    ndview<Float, 1>& get_gradient() override {
        return gradient_;
    }

    BaseMatrixOperator<Float>& get_hessian_product() override {
        return hessp_;
    }

    sycl::event update_x(const ndview<Float, 1>& x,
                         bool needHessp = false,
                         const event_vector& deps = {}) override {
        auto fill_gradient_event = fill<Float>(q_, gradient_, Float(0), deps);
        auto fill_value_event = fill<Float>(q_, tmp_, Float(0), deps);
        auto gemv_event =
            gemv(q_, A_, x, gradient_, Float(1), Float(0), { fill_gradient_event }); // Ax
        Float tmp_host = 0;
        auto xtax_event = dot_product(q_,
                                      gradient_,
                                      x,
                                      tmp_.get_mutable_data(),
                                      &tmp_host,
                                      { gemv_event }); //x^tAx
        auto btx_event =
            dot_product(q_, b_, x, tmp_.get_mutable_data(), &value_, { xtax_event }); // b^t x
        const auto kernel_minus = [=](const Float a, const Float b) -> Float {
            return a - b;
        };
        auto bias_event = element_wise(q_, kernel_minus, gradient_, b_, gradient_, { xtax_event });

        btx_event.wait_and_throw();
        value_ = -value_ + tmp_host / 2; // 1/2 x^t A x - b^t x

        return bias_event;
    }

private:
    sycl::queue q_;
    const std::int64_t n_;
    const ndview<Float, 2> A_;
    const ndview<Float, 1> b_;
    Float value_;
    ndarray<Float, 1> tmp_;
    ndarray<Float, 1> gradient_;
    LinearMatrixOperator<Float> hessp_;
};

} // namespace oneapi::dal::backend::primitives
