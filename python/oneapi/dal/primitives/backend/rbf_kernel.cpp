#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/primitives/backend/kernel_functions_py.hpp"

namespace oneapi::dal {

ONEDAL_PARAM_DISPATCH_SECTION_START(std::string, fptype) {
    ONEDAL_PARAM_DISPATCH_VALUE("float", float)
    ONEDAL_PARAM_DISPATCH_VALUE("double", double)
    ONEDAL_PARAM_DISPATCH_SECTION_END(fptype)
}

ONEDAL_PARAM_DISPATCH_SECTION_START(std::string, method) {
    ONEDAL_PARAM_DISPATCH_VALUE("dense", rbf_kernel::method::dense)
    ONEDAL_PARAM_DISPATCH_SECTION_END(method)
}

template <typename Float, typename Method, typename Task>
auto get_descriptor(const py::dict& params) {
    using namespace rbf_kernel;
    return descriptor<Float, Method, Task>{}
        .set_sigma(params["sigma"].cast<double>());
}

template <typename Caller>
struct ops_dispatcher {
    ops_dispatcher(Caller& caller)
        : caller(caller) {}

    template <typename Float, typename Method>
    auto dispatch(const py::dict& params) {
        caller.template call<Float, Method>(params);
    }

    Caller& caller;
};

OP_CALLER(compute);

template <typename Policy, typename Input, typename Result>
void init_compute_ops(py::module_& m, const std::string& name_suffix) {
    const std::string name = "compute" + name_suffix;

    m.def(name.c_str(), [](const Policy& policy,
                        const py::dict& params,
                        const table& x,
                        const table& y) {
        compute_caller<Policy, Input, Result> caller {policy, {x, y}};
        param_dispatcher_method { param_dispatcher_fptype { ops_dispatcher { caller } } }.dispatch(params);
        return caller.result;
    });
}

INSTANTIATOR(init_kernel_result);
INSTANTIATOR(init_compute_ops);

TYPE2STR(rbf_kernel::compute_result<rbf_kernel::task::compute>, "");
TYPE2STR(rbf_kernel::compute_input<rbf_kernel::task::compute>, "");
TYPE2STR(dal::detail::host_policy, "");
TYPE2STR(dal::detail::data_parallel_policy, "");

void init_rbf_kernel(py::module_& m) {
    using namespace dal::detail;
    using namespace dal::rbf_kernel;
    using input_t = compute_input<task::compute>;
    using result_t = compute_result<task::compute>;

    auto sub = m.def_submodule("rbf_kernel");
    ONEDAL_INSTANTIATE(init_kernel_result, sub, result_t);
    ONEDAL_INSTANTIATE(init_compute_ops, sub, types<host_policy, data_parallel_policy>, input_t, result_t);
}

} // namespace oneapi::dal
