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

#define PULL_COLUMN_SIGNATURE_HOST(T)                   \
    void pull_column(const default_host_policy& policy, \
                     dal::array<T>& block,              \
                     std::int64_t column_index,         \
                     const range& row_range)

#define PULL_COLUMN_SIGNATURE_DPC(T)                     \
    void pull_column(const data_parallel_policy& policy, \
                     dal::array<T>& block,               \
                     std::int64_t column_index,          \
                     const range& row_range,             \
                     sycl::usm::alloc alloc)

#define PUSH_COLUMN_SIGNATURE_HOST(T)                   \
    void push_column(const default_host_policy& policy, \
                     const dal::array<T>& block,        \
                     std::int64_t column_index,         \
                     const range& row_range)

#define PUSH_COLUMN_SIGNATURE_DPC(T)                     \
    void push_column(const data_parallel_policy& policy, \
                     const dal::array<T>& block,         \
                     std::int64_t column_index,          \
                     const range& row_range)

#define DECLARE_PULL_COLUMN_HOST(T) virtual PULL_COLUMN_SIGNATURE_HOST(T) = 0;
#define DECLARE_PULL_COLUMN_DPC(T)  virtual PULL_COLUMN_SIGNATURE_DPC(T) = 0;
#define DECLARE_PUSH_COLUMN_HOST(T) virtual PUSH_COLUMN_SIGNATURE_HOST(T) = 0;
#define DECLARE_PUSH_COLUMN_DPC(T)  virtual PUSH_COLUMN_SIGNATURE_DPC(T) = 0;

#define DEFINE_TEMPLATE_PULL_COLUMN_HOST(Derived, T)                                               \
    PULL_COLUMN_SIGNATURE_HOST(T) override {                                                       \
        static_cast<Derived*>(this)->pull_column_template(policy, block, column_index, row_range); \
    }

#define DEFINE_TEMPLATE_PULL_COLUMN_DPC(Derived, T)                     \
    PULL_COLUMN_SIGNATURE_DPC(T) override {                             \
        static_cast<Derived*>(this)->pull_column_template(policy,       \
                                                          block,        \
                                                          column_index, \
                                                          row_range,    \
                                                          alloc);       \
    }

#define DEFINE_TEMPLATE_PUSH_COLUMN_HOST(Derived, T)                                               \
    PUSH_COLUMN_SIGNATURE_HOST(T) override {                                                       \
        static_cast<Derived*>(this)->push_column_template(policy, block, column_index, row_range); \
    }

#define DEFINE_TEMPLATE_PUSH_COLUMN_DPC(Derived, T)                                                \
    PUSH_COLUMN_SIGNATURE_DPC(T) override {                                                        \
        static_cast<Derived*>(this)->push_column_template(policy, block, column_index, row_range); \
    }

class pull_column_iface {
public:
    virtual ~pull_column_iface() = default;

    DECLARE_PULL_COLUMN_HOST(float)
    DECLARE_PULL_COLUMN_HOST(double)
    DECLARE_PULL_COLUMN_HOST(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DECLARE_PULL_COLUMN_DPC(float)
    DECLARE_PULL_COLUMN_DPC(double)
    DECLARE_PULL_COLUMN_DPC(std::int32_t)
#endif
};

template <typename Derived>
class pull_column_template : public base, public pull_column_iface {
public:
    DEFINE_TEMPLATE_PULL_COLUMN_HOST(Derived, float)
    DEFINE_TEMPLATE_PULL_COLUMN_HOST(Derived, double)
    DEFINE_TEMPLATE_PULL_COLUMN_HOST(Derived, std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DEFINE_TEMPLATE_PULL_COLUMN_DPC(Derived, float)
    DEFINE_TEMPLATE_PULL_COLUMN_DPC(Derived, double)
    DEFINE_TEMPLATE_PULL_COLUMN_DPC(Derived, std::int32_t)
#endif
};

class push_column_iface {
public:
    virtual ~push_column_iface() = default;

    DECLARE_PUSH_COLUMN_HOST(float)
    DECLARE_PUSH_COLUMN_HOST(double)
    DECLARE_PUSH_COLUMN_HOST(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DECLARE_PUSH_COLUMN_DPC(float)
    DECLARE_PUSH_COLUMN_DPC(double)
    DECLARE_PUSH_COLUMN_DPC(std::int32_t)
#endif
};

template <typename Derived>
class push_column_template : public base, public push_column_iface {
public:
    DEFINE_TEMPLATE_PUSH_COLUMN_HOST(Derived, float)
    DEFINE_TEMPLATE_PUSH_COLUMN_HOST(Derived, double)
    DEFINE_TEMPLATE_PUSH_COLUMN_HOST(Derived, std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DEFINE_TEMPLATE_PUSH_COLUMN_DPC(Derived, float)
    DEFINE_TEMPLATE_PUSH_COLUMN_DPC(Derived, double)
    DEFINE_TEMPLATE_PUSH_COLUMN_DPC(Derived, std::int32_t)
#endif
};

#undef PULL_COLUMN_SIGNATURE_HOST
#undef PULL_COLUMN_SIGNATURE_DPC
#undef PUSH_COLUMN_SIGNATURE_HOST
#undef PUSH_COLUMN_SIGNATURE_DPC
#undef DECLARE_PULL_COLUMN_HOST
#undef DECLARE_PULL_COLUMN_DPC
#undef DECLARE_PUSH_COLUMN_HOST
#undef DECLARE_PUSH_COLUMN_DPC
#undef DEFINE_TEMPLATE_PULL_COLUMN_HOST
#undef DEFINE_TEMPLATE_PULL_COLUMN_DPC
#undef DEFINE_TEMPLATE_PUSH_COLUMN_HOST
#undef DEFINE_TEMPLATE_PUSH_COLUMN_DPC

} // namespace v1

using v1::pull_column_iface;
using v1::pull_column_template;
using v1::push_column_iface;
using v1::push_column_template;

} // namespace oneapi::dal::detail
