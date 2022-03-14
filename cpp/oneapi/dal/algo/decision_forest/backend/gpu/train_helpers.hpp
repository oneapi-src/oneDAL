/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dal/detail/policy.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::ext_intel_global_device_space>
        atomic_var(*ptr);
    return atomic_var.fetch_add(operand);
}

template <typename T>
inline T atomic_global_sum(T* ptr, T operand) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::ext_intel_global_device_space>
        atomic_var(*ptr);
    auto old = atomic_var.fetch_add(operand);
    return old + operand;
}

template <typename T>
inline T atomic_local_add(T* ptr, T operand) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::local_space>
        atomic_var(*ptr);
    return atomic_var.fetch_add(operand);
}

template <typename T>
inline T atomic_local_sum(T* ptr, T operand) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::local_space>
        atomic_var(*ptr);
    auto old = atomic_var.fetch_add(operand);
    return old + operand;
}

template <typename T>
inline T atomic_local_min(T* ptr, T operand) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::local_space>
        atomic_var(*ptr);
    return atomic_var.fetch_min(operand);
}

template <typename T>
inline T atomic_global_cmpxchg(T* ptr, T expected, T desired) {
    T expected_ = expected;
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     cl::sycl::access::address_space::ext_intel_global_device_space>
        atomic_var(*ptr);
    atomic_var.compare_exchange_weak(expected_,
                                     desired,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device);
    return expected_;
}

template <typename Float>
inline void merge_stat(Float& dst_count,
                       Float& dst_mean,
                       Float& dst_s2c,
                       Float src_count,
                       Float src_mean,
                       Float src_s2c) {
    if (Float(0) == src_count)
        return;

    Float sum_n1n2 = dst_count + src_count;
    Float mul_n1n2 = dst_count * src_count;
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = Float(1) / sum_n1n2;
    Float delta = src_mean - dst_mean;

    dst_s2c = dst_s2c + src_s2c + delta * delta * delta_scl;
    dst_mean = (dst_mean * dst_count + src_mean * src_count) * mean_scl;
    dst_count = sum_n1n2;
}

template <typename T, typename Index>
T* get_buf_ptr(byte_t** buf_ptr, Index elem_count) {
    T* res_ptr = reinterpret_cast<T*>(*buf_ptr);
    (*buf_ptr) += elem_count * sizeof(T);
    return res_ptr;
}

template <typename Float, typename Index = std::int32_t, typename ItemT>
inline void reduce_hist_over_group(ItemT& item,
                                   Float* slm_buf_ptr,
                                   Float& count,
                                   Float& mean,
                                   Float& sum2cent) {
    auto sbg = item.get_sub_group();
    Index sub_group_id = sbg.get_group_id();
    Index sub_group_count = item.get_local_range()[0] / sbg.get_local_range()[0];
    Index local_id = item.get_local_id(0);

    Float* count_buf_ptr = slm_buf_ptr + sub_group_count * 0;
    Float* mean_buf_ptr = slm_buf_ptr + sub_group_count * 1;
    Float* sum2cent_buf_ptr = slm_buf_ptr + sub_group_count * 2;

    if (sbg.get_local_id() == 0) {
        count_buf_ptr[sub_group_id] = count;
        mean_buf_ptr[sub_group_id] = mean;
        sum2cent_buf_ptr[sub_group_id] = sum2cent;
    }

    for (Index stride = sub_group_count / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < stride) {
            merge_stat(count_buf_ptr[local_id],
                       mean_buf_ptr[local_id],
                       sum2cent_buf_ptr[local_id],
                       count_buf_ptr[local_id + stride],
                       mean_buf_ptr[local_id + stride],
                       sum2cent_buf_ptr[local_id + stride]);
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    count = count_buf_ptr[0];
    mean = mean_buf_ptr[0];
    sum2cent = sum2cent_buf_ptr[0];
}

template <typename T, typename ItemT>
inline T reduce_min_over_group(ItemT& item, T* slm_buf, T val) {
    auto sbg = item.get_sub_group();

    //try remove this condition for perf check
    if (sbg.get_group_id() == 0 && sbg.get_local_id() == 0) {
        slm_buf[0] = val;
    }
    item.barrier(sycl::access::fence_space::local_space);

    T sbg_min = sycl::reduce_over_group(sbg, val, minimum<T>());
    if (sbg.get_local_id() == 0) {
        atomic_local_min(&slm_buf[0], sbg_min);
    }

    item.barrier(sycl::access::fence_space::local_space);

    return slm_buf[0];
}

template <typename T, typename Index, typename ItemT>
inline T reduce_min_over_group(ItemT& item, T* slm_buf, T val, Index elem_count) {
    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    slm_buf[local_id] = val;

    item.barrier(sycl::access::fence_space::local_space);
    for (Index stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            slm_buf[local_id] = sycl::min(slm_buf[local_id], slm_buf[local_id + stride]);
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    item.barrier(sycl::access::fence_space::local_space);

    return slm_buf[0];
}

template <typename T, typename ItemT>
inline T reduce_add_over_group(ItemT& item, T* slm_buf, T val) {
    auto sbg = item.get_sub_group();

    //try remove this condition for perf check
    if (sbg.get_group_id() == 0 && sbg.get_local_id() == 0) {
        slm_buf[0] = T(0);
    }
    item.barrier(sycl::access::fence_space::local_space);

    T sbg_sum = sycl::reduce_over_group(sbg, val, plus<T>());
    if (sbg.get_local_id() == 0) {
        atomic_local_add(&slm_buf[0], sbg_sum);
    }

    item.barrier(sycl::access::fence_space::local_space);

    return slm_buf[0];
}

template <typename T, typename Index, typename ItemT>
inline T reduce_add_over_group_v00(ItemT& item, T* slm_buf, T val, Index elem_count) {
    auto sbg = item.get_sub_group();
    const Index sub_group_size = sbg.get_local_range()[0];
    const Index local_id = item.get_local_id()[0];

    T count_val = val;
    slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<T>());

    Index i = elem_count;
    i = i / sub_group_size + bool(i % sub_group_size);

    for (; i > 1; i = i / sub_group_size + bool(i % sub_group_size)) {
        item.barrier(sycl::access::fence_space::local_space);

        count_val = (local_id < i) ? slm_buf[local_id] : T(0);

        item.barrier(sycl::access::fence_space::local_space);

        slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<T>());
    }

    item.barrier(sycl::access::fence_space::local_space);

    return slm_buf[0];
}

template <typename T, typename Index, typename ItemT>
inline T reduce_add_over_group_v01(ItemT& item, T* slm_buf, T val, Index elem_count) {
    auto sbg = item.get_sub_group();
    const Index sub_group_size = sbg.get_local_range()[0];
    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    T count_val = val;

    for (Index i = local_size; i > 1; i = i / sub_group_size) {
        slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<T>());

        item.barrier(sycl::access::fence_space::local_space);

        count_val = (local_id < i) ? slm_buf[local_id] : T(0);

        item.barrier(sycl::access::fence_space::local_space);

        slm_buf[sbg.get_group_id()] = sycl::reduce_over_group(sbg, count_val, plus<T>());

        item.barrier(sycl::access::fence_space::local_space);
    }

    return slm_buf[0];
}

template <typename T, typename Index, typename ItemT>
inline T reduce_add_over_group_v02(ItemT& item, T* slm_buf, T val, Index elem_count) {
    return sycl::reduce_over_group(item.get_group(), val, plus<T>());
}

template <typename T, typename Index, typename ItemT>
inline T reduce_add_over_group(ItemT& item, T* slm_buf, T val, Index elem_count) {
    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    slm_buf[local_id] = val;

    item.barrier(sycl::access::fence_space::local_space);
    for (Index stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            slm_buf[local_id] += slm_buf[local_id + stride];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    item.barrier(sycl::access::fence_space::local_space);

    return slm_buf[0];
}
} // namespace oneapi::dal::decision_forest::backend

#endif
