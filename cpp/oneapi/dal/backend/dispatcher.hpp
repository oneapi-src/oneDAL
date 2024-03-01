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

#include "oneapi/dal/global_context.hpp"

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/detail/cpu_info.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/dispatcher_cpu.hpp"

#define KERNEL_SPEC(spec, ...) ::oneapi::dal::backend::kernel_spec<spec, __VA_ARGS__>

#define KERNEL_SINGLE_NODE_CPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::single_node_cpu_kernel, __VA_ARGS__)

#define KERNEL_SINGLE_NODE_GPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::single_node_gpu_kernel, __VA_ARGS__)

#define KERNEL_UNIVERSAL_SPMD_GPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::universal_spmd_gpu_kernel, __VA_ARGS__)

namespace oneapi::dal::backend {

struct cpu_dispatch_sse2 {};
struct cpu_dispatch_sse42 {};
struct cpu_dispatch_avx2 {};
struct cpu_dispatch_avx512 {};

using cpu_dispatch_default = cpu_dispatch_sse2;

#define __CPU_TAG_SSE2__    oneapi::dal::backend::cpu_dispatch_sse2
#define __CPU_TAG_SSE42__   oneapi::dal::backend::cpu_dispatch_sse42
#define __CPU_TAG_AVX2__    oneapi::dal::backend::cpu_dispatch_avx2
#define __CPU_TAG_AVX512__  oneapi::dal::backend::cpu_dispatch_avx512
#define __CPU_TAG_DEFAULT__ oneapi::dal::backend::cpu_dispatch_default

template <typename MemoryAccessKind>
class communicator_provider : public base {
public:
    communicator_provider() = default;
    communicator_provider(const communicator<MemoryAccessKind>& comm)
            : comm_(new communicator<MemoryAccessKind>{ comm }) {}

    const communicator<MemoryAccessKind>& get_communicator() const {
        if (!comm_) {
            comm_.reset(new communicator<MemoryAccessKind>{});
        }
        return *comm_;
    }

private:
    mutable std::unique_ptr<communicator<MemoryAccessKind>> comm_;
};

class context_cpu : public communicator_provider<spmd::device_memory_access::none> {
public:
    explicit context_cpu(const detail::host_policy& policy = detail::host_policy::get_default())
            : cpu_extensions_(policy.get_enabled_cpu_extensions()) {
        global_context::get_global_context();
    }

    explicit context_cpu(const detail::spmd_host_policy& policy)
            : communicator_provider<spmd::device_memory_access::none>(policy.get_communicator()),
              cpu_extensions_(policy.get_local().get_enabled_cpu_extensions()) {
        global_context::get_global_context();
    }

    explicit context_cpu(const spmd::communicator<spmd::device_memory_access::none>& comm)
            : communicator_provider<spmd::device_memory_access::none>(comm),
              cpu_extensions_(detail::host_policy::get_default().get_enabled_cpu_extensions()) {}

    detail::cpu_extension get_enabled_cpu_extensions() const {
        return cpu_extensions_;
    }

private:
    detail::cpu_extension cpu_extensions_;
};

#ifdef ONEDAL_DATA_PARALLEL
class context_gpu : public communicator_provider<spmd::device_memory_access::usm> {
public:
    explicit context_gpu(const detail::data_parallel_policy& policy) : queue_(policy.get_queue()) {}

    explicit context_gpu(const detail::spmd_data_parallel_policy& policy)
            : communicator_provider(policy.get_communicator()),
              queue_(policy.get_local().get_queue()) {}

    sycl::queue& get_queue() const {
        return queue_;
    }

private:
    sycl::queue& queue_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <typename CpuBranch, typename GpuBranch>
inline auto dispatch_by_device(const detail::data_parallel_policy& policy,
                               CpuBranch&& cpu_branch,
                               GpuBranch&& gpu_branch) {
    const auto device = policy.get_queue().get_device();
    if (device.is_cpu()) {
        return cpu_branch();
    }
    else if (device.is_gpu()) {
        return gpu_branch();
    }
    else {
        throw unsupported_device{ dal::detail::error_messages::unsupported_device_type() };
    }
}
#endif

/// Tag that indicates CPU kernel for single-node
struct single_node_cpu_kernel {};

/// Tag that indicates GPU kernel for single-node
struct single_node_gpu_kernel {};

/// Tag that indicates universal GPU kernel for single-node and SPMD modes
struct universal_spmd_gpu_kernel {};

template <typename Tag, typename Kernel>
struct kernel_spec {};

template <typename... KernelSpecs>
struct kernel_dispatcher {};

template <typename Context, typename Kernel, typename... Args>
struct kernel_return_type {
    using type = decltype(std::declval<Kernel>()(std::declval<Context>(), std::declval<Args>()...));
};

template <typename... Args>
using cpu_kernel_return_t = typename kernel_return_type<context_cpu, Args...>::type;

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
using gpu_kernel_return_t = typename kernel_return_type<context_gpu, Args...>::type;
#endif

/// Dispatcher for the case of only CPU and single-node algorithm
template <typename CpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::host_policy& policy, Args&&... args) const {
        return CpuKernel{}(context_cpu{ policy }, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto operator()(const detail::spmd_host_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() -> cpu_kernel_return_t<CpuKernel, Args...> {
                // We have to specify return type for this lambda as compiler cannot
                // infer it from a body that consist of single `throw` expression
                using msg = detail::error_messages;
                throw unimplemented{ msg::algorithm_is_not_implemented_for_this_device() };
            });
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }
#endif
};

#ifdef ONEDAL_DATA_PARALLEL
/// Dispatcher for the case of single-node CPU and GPU algorithm
template <typename CpuKernel, typename GpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>,
                         kernel_spec<single_node_gpu_kernel, GpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
/// Dispatcher for the case of single-node CPU and multi-node
/// GPU algorithm based on universal SPMD kernel
template <typename CpuKernel, typename GpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>,
                         kernel_spec<universal_spmd_gpu_kernel, GpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy.get_local(),
            [&]() -> gpu_kernel_return_t<GpuKernel, Args...> {
                // We have to specify return type for this lambda as compiler cannot
                // infer it from a body that consist of single `throw` expression
                using msg = detail::error_messages;
                throw unimplemented{
                    msg::spmd_version_of_algorithm_is_not_implemented_for_this_device()
                };
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }
};
#endif

inline bool test_cpu_extension(detail::cpu_extension mask, detail::cpu_extension test) {
    return mask >= test;
}

template <typename Op>
inline constexpr auto dispatch_by_cpu(const context_cpu& ctx, Op&& op) {
    [[maybe_unused]] const detail::cpu_extension cpu_ex = ctx.get_enabled_cpu_extensions();
    ONEDAL_IF_CPU_DISPATCH_AVX512(if (test_cpu_extension(cpu_ex, cpu_extension::avx512)) {
        return op(cpu_dispatch_avx512{});
    })
    ONEDAL_IF_CPU_DISPATCH_AVX2(
        if (test_cpu_extension(cpu_ex, detail::cpu_extension::avx2)) { return op(cpu_dispatch_avx2{}); })
    ONEDAL_IF_CPU_DISPATCH_SSE42(
        if (test_cpu_extension(cpu_ex, detail::cpu_extension::sse42)) { return op(cpu_dispatch_sse42{}); })
    return op(cpu_dispatch_default{});
}

template <typename Op, typename OnUnknown>
inline constexpr auto dispatch_by_data_type(data_type dtype, Op&& op, OnUnknown&& on_unknown) {
    switch (dtype) {
        case data_type::int8: return op(std::int8_t{});
        case data_type::uint8: return op(std::uint8_t{});
        case data_type::int16: return op(std::int16_t{});
        case data_type::uint16: return op(std::uint16_t{});
        case data_type::int32: return op(std::int32_t{});
        case data_type::uint32: return op(std::uint32_t{});
        case data_type::int64: return op(std::int64_t{});
        case data_type::uint64: return op(std::uint64_t{});
        case data_type::float32: return op(float{});
        case data_type::float64: return op(double{});
        default: return on_unknown(dtype);
    }
}

template <typename Op, typename ResultType = std::invoke_result_t<Op, float>>
inline constexpr ResultType dispatch_by_data_type(data_type dtype, Op&& op) {
    // Necessary to make the return type conformant with
    // other dispatch branches
    const auto on_unknown = [](data_type) -> ResultType {
        using msg = dal::detail::error_messages;
        throw unimplemented{ msg::unsupported_conversion_types() };
    };

    return dispatch_by_data_type(dtype, std::forward<Op>(op), on_unknown);
}

namespace impl {

template <typename Result, typename... Types>
struct type_holder {
    using result_t = Result;

    template <typename Tail>
    using add_tail = type_holder<Result, Types..., Tail>;

    template <typename Op>
    constexpr static inline Result evaluate(Op&& op) {
        return op(Types{}...);
    }
};

template <typename TypeHolder, typename Op>
inline constexpr auto multi_dispatch_by_data_type(Op&& op) {
    return TypeHolder::evaluate(std::forward<Op>(op));
}

template <typename TypeHolder, typename Op, typename Head, typename... Tail>
inline constexpr auto multi_dispatch_by_data_type(Op&& op, Head&& head, Tail&&... tail) {
    using result_t = typename TypeHolder::result_t;
    const auto functor = [&](auto arg) -> result_t {
        using type_t = std::decay_t<decltype(arg)>;
        using holder_t = typename TypeHolder::template add_tail<type_t>;
        return multi_dispatch_by_data_type<holder_t>( //
            std::forward<Op>(op),
            std::forward<Tail>(tail)...);
    };
    return dispatch_by_data_type(head, functor);
}

template <std::size_t n, typename DefaultType, typename Op, typename... Types>
struct invoke_result_multiple_impl {
    using next_t = invoke_result_multiple_impl<n - 1, DefaultType, Op, DefaultType, Types...>;
    using type = typename next_t::type;
};

template <typename DefaultType, typename Op, typename... Types>
struct invoke_result_multiple_impl<0ul, DefaultType, Op, Types...> {
    using type = std::invoke_result_t<Op, Types...>;
};

template <typename Op, std::size_t n, typename DefaultType = float>
using invoke_result_multiple_t = typename invoke_result_multiple_impl<n, DefaultType, Op>::type;

} // namespace impl

// Signature of this function is slightly different from
// a simple `dispatch_by_data_type` due to inconsistency
// with a `std::visit` which it heavily resembles
template <typename ResultType, typename Op, typename... Types>
inline constexpr ResultType multi_dispatch_by_data_type(Op&& op, Types&&... types) {
    using holder_t = impl::type_holder<ResultType>;
    return impl::multi_dispatch_by_data_type<holder_t, Op>( //
        std::forward<Op>(op),
        std::forward<Types>(types)...);
}

template <typename Op, typename... Types>
inline constexpr auto multi_dispatch_by_data_type(Op&& op, Types&&... types) {
    using result_t = impl::invoke_result_multiple_t<Op, sizeof...(Types), float>;
    return multi_dispatch_by_data_type<result_t, Op>( //
        std::forward<Op>(op),
        std::forward<Types>(types)...);
}

} // namespace oneapi::dal::backend
