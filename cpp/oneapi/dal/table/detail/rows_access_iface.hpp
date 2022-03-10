/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::detail {
namespace v1 {

#define PULL_ROWS_SIGNATURE_HOST(T) \
    void pull_rows(const default_host_policy& policy, dal::array<T>& block, const range& row_range)

#define PULL_ROWS_SIGNATURE_DPC(T)                     \
    void pull_rows(const data_parallel_policy& policy, \
                   dal::array<T>& block,               \
                   const range& row_range,             \
                   sycl::usm::alloc alloc)

#define PUSH_ROWS_SIGNATURE_HOST(T)                   \
    void push_rows(const default_host_policy& policy, \
                   const dal::array<T>& block,        \
                   const range& row_range)

#define PUSH_ROWS_SIGNATURE_DPC(T)                     \
    void push_rows(const data_parallel_policy& policy, \
                   const dal::array<T>& block,         \
                   const range& row_range)

#define DECLARE_PULL_ROWS_HOST(T) virtual PULL_ROWS_SIGNATURE_HOST(T) = 0;
#define DECLARE_PULL_ROWS_DPC(T)  virtual PULL_ROWS_SIGNATURE_DPC(T) = 0;
#define DECLARE_PUSH_ROWS_HOST(T) virtual PUSH_ROWS_SIGNATURE_HOST(T) = 0;
#define DECLARE_PUSH_ROWS_DPC(T)  virtual PUSH_ROWS_SIGNATURE_DPC(T) = 0;

#define DEFINE_TEMPLATE_PULL_ROWS_HOST(Derived, T)                                 \
    PULL_ROWS_SIGNATURE_HOST(T) override {                                         \
        static_cast<Derived*>(this)->pull_rows_template(policy, block, row_range); \
    }

#define DEFINE_TEMPLATE_PULL_ROWS_DPC(Derived, T)                                         \
    PULL_ROWS_SIGNATURE_DPC(T) override {                                                 \
        static_cast<Derived*>(this)->pull_rows_template(policy, block, row_range, alloc); \
    }

#define DEFINE_TEMPLATE_PUSH_ROWS_HOST(Derived, T)                                 \
    PUSH_ROWS_SIGNATURE_HOST(T) override {                                         \
        static_cast<Derived*>(this)->push_rows_template(policy, block, row_range); \
    }

#define DEFINE_TEMPLATE_PUSH_ROWS_DPC(Derived, T)                                  \
    PUSH_ROWS_SIGNATURE_DPC(T) override {                                          \
        static_cast<Derived*>(this)->push_rows_template(policy, block, row_range); \
    }

class pull_rows_iface {
public:
    virtual ~pull_rows_iface() = default;

    DECLARE_PULL_ROWS_HOST(float)
    DECLARE_PULL_ROWS_HOST(double)
    DECLARE_PULL_ROWS_HOST(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DECLARE_PULL_ROWS_DPC(float)
    DECLARE_PULL_ROWS_DPC(double)
    DECLARE_PULL_ROWS_DPC(std::int32_t)
#endif
};

template <typename Derived>
class pull_rows_template : public base, public pull_rows_iface {
public:
    DEFINE_TEMPLATE_PULL_ROWS_HOST(Derived, float)
    DEFINE_TEMPLATE_PULL_ROWS_HOST(Derived, double)
    DEFINE_TEMPLATE_PULL_ROWS_HOST(Derived, std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DEFINE_TEMPLATE_PULL_ROWS_DPC(Derived, float)
    DEFINE_TEMPLATE_PULL_ROWS_DPC(Derived, double)
    DEFINE_TEMPLATE_PULL_ROWS_DPC(Derived, std::int32_t)
#endif
};

class push_rows_iface {
public:
    virtual ~push_rows_iface() = default;

    DECLARE_PUSH_ROWS_HOST(float)
    DECLARE_PUSH_ROWS_HOST(double)
    DECLARE_PUSH_ROWS_HOST(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DECLARE_PUSH_ROWS_DPC(float)
    DECLARE_PUSH_ROWS_DPC(double)
    DECLARE_PUSH_ROWS_DPC(std::int32_t)
#endif
};

template <typename Derived>
class push_rows_template : public base, public push_rows_iface {
public:
    DEFINE_TEMPLATE_PUSH_ROWS_HOST(Derived, float)
    DEFINE_TEMPLATE_PUSH_ROWS_HOST(Derived, double)
    DEFINE_TEMPLATE_PUSH_ROWS_HOST(Derived, std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DEFINE_TEMPLATE_PUSH_ROWS_DPC(Derived, float)
    DEFINE_TEMPLATE_PUSH_ROWS_DPC(Derived, double)
    DEFINE_TEMPLATE_PUSH_ROWS_DPC(Derived, std::int32_t)
#endif
};

#undef PULL_ROWS_SIGNATURE_HOST
#undef PULL_ROWS_SIGNATURE_DPC
#undef PUSH_ROWS_SIGNATURE_HOST
#undef PUSH_ROWS_SIGNATURE_DPC
#undef DECLARE_PULL_ROWS_HOST
#undef DECLARE_PULL_ROWS_DPC
#undef DECLARE_PUSH_ROWS_HOST
#undef DECLARE_PUSH_ROWS_DPC
#undef DEFINE_TEMPLATE_PULL_ROWS_HOST
#undef DEFINE_TEMPLATE_PULL_ROWS_DPC
#undef DEFINE_TEMPLATE_PUSH_ROWS_HOST
#undef DEFINE_TEMPLATE_PUSH_ROWS_DPC

} // namespace v1

using v1::pull_rows_iface;
using v1::pull_rows_template;
using v1::push_rows_iface;
using v1::push_rows_template;

} // namespace oneapi::dal::detail
