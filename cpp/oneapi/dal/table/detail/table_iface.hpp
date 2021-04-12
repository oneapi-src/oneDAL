/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal {
namespace v1 {
class table_metadata;
enum class data_layout;
enum class feature_type;
} // namespace v1

using v1::table_metadata;
using v1::data_layout;
using v1::feature_type;

} // namespace oneapi::dal

namespace oneapi::dal::detail {

#define DECLARE_PULL_ROWS_HOST(T) \
    virtual void pull_rows(const default_host_policy&, array<T>&, const range&) = 0;

#define DECLARE_PULL_ROWS_DPC(T)                        \
    virtual void pull_rows(const data_parallel_policy&, \
                           array<T>&,                   \
                           const range&,                \
                           sycl::usm::alloc) = 0;

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_PULL_ROWS(T)  \
    DECLARE_PULL_ROWS_HOST(T) \
    DECLARE_PULL_ROWS_DPC(T)
#else
#define DECLARE_PULL_ROWS(T) DECLARE_PULL_ROWS_HOST(T)
#endif

class pull_rows_iface {
public:
    virtual ~pull_rows_iface() = default;

    DECLARE_PULL_ROWS(float)
    DECLARE_PULL_ROWS(double)
    DECLARE_PULL_ROWS(std::int32_t)
};

#undef DECLARE_PULL_ROWS_HOST
#undef DECLARE_PULL_ROWS_DPC
#undef DECLARE_PULL_ROWS

#define DECLARE_PUSH_ROWS_HOST(T) \
    virtual void push_rows(const default_host_policy&, const array<T>&, const range&) = 0;

#define DECLARE_PUSH_ROWS_DPC(T) \
    virtual void push_rows(const data_parallel_policy&, const array<T>&, const range&) = 0;

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_PUSH_ROWS(T)  \
    DECLARE_PUSH_ROWS_HOST(T) \
    DECLARE_PUSH_ROWS_DPC(T)
#else
#define DECLARE_PUSH_ROWS(T) DECLARE_PUSH_ROWS_HOST(T)
#endif

class push_rows_iface {
public:
    virtual ~push_rows_iface() = default;

    DECLARE_PUSH_ROWS(float)
    DECLARE_PUSH_ROWS(double)
    DECLARE_PUSH_ROWS(std::int32_t)
};

#undef DECLARE_PUSH_ROWS_HOST
#undef DECLARE_PUSH_ROWS_DPC
#undef DECLARE_PUSH_ROWS

#define DECLARE_PULL_COLUMN_HOST(T) \
    virtual void pull_column(const default_host_policy&, array<T>&, std::int64_t, const range&) = 0;

#define DECLARE_PULL_COLUMN_DPC(T)                        \
    virtual void pull_column(const data_parallel_policy&, \
                             array<T>&,                   \
                             std::int64_t,                \
                             const range&,                \
                             sycl::usm::alloc) = 0;

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_PULL_COLUMN(T)  \
    DECLARE_PULL_COLUMN_HOST(T) \
    DECLARE_PULL_COLUMN_DPC(T)
#else
#define DECLARE_PULL_COLUMN(T) DECLARE_PULL_COLUMN_HOST(T)
#endif

class pull_column_iface {
public:
    virtual ~pull_column_iface() = default;

    DECLARE_PULL_COLUMN(float)
    DECLARE_PULL_COLUMN(double)
    DECLARE_PULL_COLUMN(std::int32_t)
};

#undef DECLARE_PULL_COLUMN_HOST
#undef DECLARE_PULL_COLUMN_DPC
#undef DECLARE_PULL_COLUMN

#define DECLARE_PUSH_COLUMN_HOST(T)                      \
    virtual void push_column(const default_host_policy&, \
                             const array<T>&,            \
                             std::int64_t,               \
                             const range&) = 0;

#define DECLARE_PUSH_COLUMN_DPC(T)                        \
    virtual void push_column(const data_parallel_policy&, \
                             const array<T>&,             \
                             std::int64_t,                \
                             const range&) = 0;

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_PUSH_COLUMN(T)  \
    DECLARE_PUSH_COLUMN_HOST(T) \
    DECLARE_PUSH_COLUMN_DPC(T)
#else
#define DECLARE_PUSH_COLUMN(T) DECLARE_PUSH_COLUMN_HOST(T)
#endif

class push_column_iface {
public:
    virtual ~push_column_iface() = default;

    DECLARE_PUSH_COLUMN(float)
    DECLARE_PUSH_COLUMN(double)
    DECLARE_PUSH_COLUMN(std::int32_t)
};

#undef DECLARE_PUSH_COLUMN_HOST
#undef DECLARE_PUSH_COLUMN_DPC
#undef DECLARE_PUSH_COLUMN

class table_iface {
public:
    virtual ~table_iface() = default;
    virtual std::int64_t get_column_count() const = 0;
    virtual std::int64_t get_row_count() const = 0;
    virtual std::int64_t get_kind() const = 0;
    virtual data_layout get_data_layout() const = 0;
    virtual const table_metadata& get_metadata() const = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
};

class homogen_table_iface : public table_iface {
public:
    virtual const void* get_data() const = 0;
};

class table_builder_iface {
public:
    virtual ~table_builder_iface() = default;
    virtual table_iface* build() = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
    virtual push_rows_iface* get_push_rows_iface() = 0;
    virtual push_column_iface* get_push_column_iface() = 0;
};

class homogen_table_builder_iface : public table_builder_iface {
public:
    virtual homogen_table_iface* build_homogen() = 0;

    virtual void set_data_type(data_type dt) = 0;
    virtual void set_layout(data_layout layout) = 0;
    virtual void set_feature_type(feature_type ft) = 0;

    virtual void reset(homogen_table_iface& t) = 0;
    virtual void reset(const array<byte_t>& data,
                       std::int64_t row_count,
                       std::int64_t column_count) = 0;

    virtual void allocate(std::int64_t row_count, //
                          std::int64_t column_count) = 0;

    virtual void copy_data(const void* data, //
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual void allocate(const sycl::queue& queue,
                          std::int64_t row_count,
                          std::int64_t column_count,
                          sycl::usm::alloc kind) = 0;

    virtual void copy_data(sycl::queue& queue,
                           const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;
#endif
};

template <typename Derived>
class pull_rows_template : public pull_rows_iface {
public:
    void pull_rows(const default_host_policy& policy,
                   array<float>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows);
    }

    void pull_rows(const default_host_policy& policy,
                   array<double>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows);
    }

    void pull_rows(const default_host_policy& policy,
                   array<std::int32_t>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void pull_rows(const data_parallel_policy& policy,
                   array<float>& block,
                   const range& rows,
                   sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows, alloc);
    }

    void pull_rows(const data_parallel_policy& policy,
                   array<double>& block,
                   const range& rows,
                   sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows, alloc);
    }

    void pull_rows(const data_parallel_policy& policy,
                   array<std::int32_t>& block,
                   const range& rows,
                   sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_rows(policy, block, rows, alloc);
    }
#endif
};

template <typename Derived>
class pull_column_template : public pull_column_iface {
public:
    void pull_column(const default_host_policy& policy,
                     array<float>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows);
    }

    void pull_column(const default_host_policy& policy,
                     array<double>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows);
    }

    void pull_column(const default_host_policy& policy,
                     array<std::int32_t>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void pull_column(const data_parallel_policy& policy,
                     array<float>& block,
                     std::int64_t column_index,
                     const range& rows,
                     sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows, alloc);
    }

    void pull_column(const data_parallel_policy& policy,
                     array<double>& block,
                     std::int64_t column_index,
                     const range& rows,
                     sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows, alloc);
    }

    void pull_column(const data_parallel_policy& policy,
                     array<std::int32_t>& block,
                     std::int64_t column_index,
                     const range& rows,
                     sycl::usm::alloc alloc) override {
        static_cast<Derived*>(this)->pull_column(policy, block, column_index, rows, alloc);
    }
#endif
};

template <typename Derived>
class push_rows_template : public push_rows_iface {
public:
    void push_rows(const default_host_policy& policy,
                   const array<float>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }

    void push_rows(const default_host_policy& policy,
                   const array<double>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }

    void push_rows(const default_host_policy& policy,
                   const array<std::int32_t>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void push_rows(const data_parallel_policy& policy,
                   const array<float>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }

    void push_rows(const data_parallel_policy& policy,
                   const array<double>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }

    void push_rows(const data_parallel_policy& policy,
                   const array<std::int32_t>& block,
                   const range& rows) override {
        static_cast<Derived*>(this)->push_rows(policy, block, rows);
    }
#endif
};

template <typename Derived>
class push_column_template : public push_column_iface {
public:
    void push_column(const default_host_policy& policy,
                     const array<float>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }

    void push_column(const default_host_policy& policy,
                     const array<double>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }

    void push_column(const default_host_policy& policy,
                     const array<std::int32_t>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void push_column(const data_parallel_policy& policy,
                     const array<float>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }

    void push_column(const data_parallel_policy& policy,
                     const array<double>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }

    void push_column(const data_parallel_policy& policy,
                     const array<std::int32_t>& block,
                     std::int64_t column_index,
                     const range& rows) override {
        static_cast<Derived*>(this)->push_column(policy, block, column_index, rows);
    }
#endif
};

template <typename Object>
inline std::shared_ptr<pull_rows_iface> get_pull_rows_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_rows_iface>{ pimpl, pimpl->get_pull_rows_iface() };
}

template <typename Object>
inline std::shared_ptr<pull_column_iface> get_pull_column_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_column_iface>{ pimpl, pimpl->get_pull_column_iface() };
}

template <typename Object>
inline std::shared_ptr<push_rows_iface> get_push_rows_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<push_rows_iface>{ pimpl, pimpl->get_push_rows_iface() };
}

template <typename Object>
inline std::shared_ptr<push_column_iface> get_push_column_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<push_column_iface>{ pimpl, pimpl->get_push_column_iface() };
}

} // namespace oneapi::dal::detail
