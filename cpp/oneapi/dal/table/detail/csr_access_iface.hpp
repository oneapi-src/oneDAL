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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal {
namespace v1 {
enum class sparse_indexing;
} // namespace v1

using v1::sparse_indexing;

} // namespace oneapi::dal

namespace oneapi::dal::detail {
namespace v1 {

#define PULL_CSR_BLOCK_SIGNATURE_HOST(T)                          \
    void pull_csr_block(const default_host_policy& policy,        \
                        dal::array<T>& data,                      \
                        dal::array<std::int64_t>& column_indices, \
                        dal::array<std::int64_t>& row_offsets,    \
                        const dal::sparse_indexing& indexing,     \
                        const range& row_range)

#define PULL_CSR_BLOCK_SIGNATURE_DPC(T)                           \
    void pull_csr_block(const data_parallel_policy& policy,       \
                        dal::array<T>& data,                      \
                        dal::array<std::int64_t>& column_indices, \
                        dal::array<std::int64_t>& row_offsets,    \
                        const dal::sparse_indexing& indexing,     \
                        const range& row_range,                   \
                        sycl::usm::alloc alloc)

#define DECLARE_PULL_CSR_BLOCK_HOST(T) virtual PULL_CSR_BLOCK_SIGNATURE_HOST(T) = 0;
#define DECLARE_PULL_CSR_BLOCK_DPC(T)  virtual PULL_CSR_BLOCK_SIGNATURE_DPC(T) = 0;

#define DEFINE_TEMPLATE_PULL_CSR_BLOCK_HOST(Derived, T)                      \
    PULL_CSR_BLOCK_SIGNATURE_HOST(T) override {                              \
        static_cast<Derived*>(this)->pull_csr_block_template(policy,         \
                                                             data,           \
                                                             column_indices, \
                                                             row_offsets,    \
                                                             indexing,       \
                                                             row_range);     \
    }

#define DEFINE_TEMPLATE_PULL_CSR_BLOCK_DPC(Derived, T)                       \
    PULL_CSR_BLOCK_SIGNATURE_DPC(T) override {                               \
        static_cast<Derived*>(this)->pull_csr_block_template(policy,         \
                                                             data,           \
                                                             column_indices, \
                                                             row_offsets,    \
                                                             indexing,       \
                                                             row_range,      \
                                                             alloc);         \
    }

class pull_csr_block_iface {
public:
    virtual ~pull_csr_block_iface() = default;

    DECLARE_PULL_CSR_BLOCK_HOST(float)
    DECLARE_PULL_CSR_BLOCK_HOST(double)
    DECLARE_PULL_CSR_BLOCK_HOST(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DECLARE_PULL_CSR_BLOCK_DPC(float)
    DECLARE_PULL_CSR_BLOCK_DPC(double)
    DECLARE_PULL_CSR_BLOCK_DPC(std::int32_t)
#endif
};

template <typename Derived>
class pull_csr_block_template : public base, public pull_csr_block_iface {
public:
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_HOST(Derived, float)
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_HOST(Derived, double)
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_HOST(Derived, std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_DPC(Derived, float)
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_DPC(Derived, double)
    DEFINE_TEMPLATE_PULL_CSR_BLOCK_DPC(Derived, std::int32_t)
#endif
};

#undef PULL_CSR_BLOCK_SIGNATURE_HOST
#undef DECLARE_PULL_CSR_BLOCK_HOST
#undef DEFINE_TEMPLATE_PULL_CSR_BLOCK_HOST
#undef PULL_CSR_BLOCK_SIGNATURE_DPC
#undef DECLARE_PULL_CSR_BLOCK_DPC
#undef DEFINE_TEMPLATE_PULL_CSR_BLOCK_DPC

} // namespace v1

using v1::pull_csr_block_iface;
using v1::pull_csr_block_template;

} // namespace oneapi::dal::detail
