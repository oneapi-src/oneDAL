#include "oneapi/dal/common/backend/type_utils_py.hpp"
#include "oneapi/dal/common/backend/serialization_py.hpp"

#include "oneapi/dal/algo/svm.hpp"

namespace py = pybind11;

namespace oneapi::dal {

ONEDAL_PARAM_DISPATCH_SECTION_START(std::string, fptype) {
    ONEDAL_PARAM_DISPATCH_VALUE("float", float)
    ONEDAL_PARAM_DISPATCH_VALUE("double", double)
    ONEDAL_PARAM_DISPATCH_SECTION_END(fptype)
}

ONEDAL_PARAM_DISPATCH_SECTION_START(std::string, method) {
    ONEDAL_PARAM_DISPATCH_VALUE("thunder", svm::method::thunder)
    ONEDAL_PARAM_DISPATCH_VALUE("smo", svm::method::smo)
    ONEDAL_PARAM_DISPATCH_SECTION_END(method)
}

template <typename Kernel>
auto get_kernel_descriptor(const py::dict& params) {
    using float_t = typename Kernel::float_t;
    auto kernel = Kernel{};
    if constexpr (std::is_same_v<Kernel, linear_kernel::descriptor<float_t>>) {
        kernel.set_scale(params["scale"].cast<double>())
              .set_shift(params["shift"].cast<double>());
    }
    if constexpr (std::is_same_v<Kernel, polynomial_kernel::descriptor<float_t>>) {
        kernel.set_scale(params["scale"].cast<double>())
              .set_shift(params["shift"].cast<double>())
              .set_degree(params["degree"].cast<std::int64_t>());
    }
    if constexpr (std::is_same_v<Kernel, rbf_kernel::descriptor<float_t>>) {
        kernel.set_sigma(params["sigma"].cast<double>());
    }
    return kernel;
}

template <typename Float, typename Method, typename Task, typename Kernel>
auto get_descriptor(const py::dict& params) {
    using namespace svm;
    constexpr bool is_cls = std::is_same_v<Task, task::classification>;
    constexpr bool is_nu_cls = std::is_same_v<Task, task::nu_classification>;
    constexpr bool is_reg = std::is_same_v<Task, task::regression>;
    constexpr bool is_nu_reg = std::is_same_v<Task, task::nu_regression>;

    auto desc = descriptor<Float, Method, Task, Kernel>{ get_kernel_descriptor<Kernel>(params) }
        .set_max_iteration_count(params["max_iteration_count"].cast<std::int64_t>())
        .set_accuracy_threshold(params["accuracy_threshold"].cast<double>())
        .set_cache_size(params["cache_size"].cast<double>())
        .set_tau(params["tau"].cast<double>())
        .set_shrinking(params["shrinking"].cast<bool>());

    if constexpr(is_cls || is_reg || is_nu_reg) {
        desc.set_c(params["c"].cast<double>());
    }
    if constexpr(is_cls || is_nu_cls) {
        desc.set_class_count(params["class_count"].cast<std::int64_t>());
    }
    if constexpr(is_reg) {
        desc.set_epsilon(params["epsilon"].cast<double>());
    }
    if constexpr(is_nu_reg || is_nu_cls) {
        desc.set_nu(params["nu"].cast<double>());
    }

    return desc;
}

template <typename Caller>
struct ops_dispatcher {
    ops_dispatcher(Caller& caller)
        : caller(caller) {}

    template <typename Float, typename Method>
    auto dispatch(const py::dict& params) {
        if constexpr (!svm::detail::is_valid_method_task_combination<Method, typename Caller::task_t> &&
                      !svm::detail::is_valid_method_nu_task_combination<Method, typename Caller::task_t>) {
            auto kernel = params["kernel"].cast<std::string>();
            if (kernel == "linear") {
                caller.template call<Float, Method, linear_kernel::descriptor<Float>>(params);
            } else if (kernel == "rbf") {
                caller.template call<Float, Method, rbf_kernel::descriptor<Float>>(params);
            } else if (kernel == "poly") {
                caller.template call<Float, Method, polynomial_kernel::descriptor<Float>>(params);
            } else {
                throw std::runtime_error("Invalid value for parameter <kernel>");
            }
        } /*else ONEDAL_ASSERT()*/
    }

    Caller& caller;
};

OP_CALLER(train);
OP_CALLER(infer);

template <typename Policy, typename Task>
void init_train_ops(py::module_& m, const std::string& name_suffix) {
    const std::string name = "train" + name_suffix;

    m.def(name.c_str(), [](const Policy& policy,
                      const py::dict& params,
                      const table& data,
                      const table& labels,
                      const table& weights) {
        using input_t = svm::train_input<Task>;
        using result_t = svm::train_result<Task>;

        train_caller<Policy, input_t, result_t> caller {policy, {data, labels, weights}};
        param_dispatcher_method { param_dispatcher_fptype { ops_dispatcher { caller } } }.dispatch(params);
        return caller.result;
    });
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m, const std::string& name_suffix) {
    const std::string name = "infer" + name_suffix;

    m.def(name.c_str(), [](const Policy& policy,
                      const py::dict& params,
                      const svm::model<Task>& model,
                      const table& data) {
        using input_t = svm::infer_input<Task>;
        using result_t = svm::infer_result<Task>;

        infer_caller<Policy, input_t, result_t> caller {policy, {model, data}};
        param_dispatcher_method { param_dispatcher_fptype { ops_dispatcher { caller } } }.dispatch(params);
        return caller.result;
    });
}

template <typename Task>
void init_model(py::module_& m, const std::string& name_suffix) {
    using namespace svm;
    using model_t = model<Task>;

    const std::string name = "model" + name_suffix;

    auto cls = py::class_<model_t>(m, name.c_str())
        .def(py::init())
        .def(py::pickle(
            [](const model_t& m) { return serialize(m); },
            [](const py::bytes& bytes) { return deserialize<model_t>(bytes); }))
        .def_property_readonly("support_vector_count", &model_t::get_support_vector_count)
        .DEF_ONEDAL_PROPERTY(support_vectors, model_t)
        .DEF_ONEDAL_PROPERTY(coeffs, model_t)
        .DEF_ONEDAL_PROPERTY(biases, model_t);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;
    constexpr bool is_nu_classification = std::is_same_v<Task, task::nu_classification>;

    if constexpr (is_classification || is_nu_classification) {
        cls.def_property("first_class_label", &model_t::get_first_class_label, &model_t::template set_first_class_label<>);
        cls.def_property("second_class_label", &model_t::get_second_class_label, &model_t::template set_second_class_label<>);
    }
}

template <typename Task>
void init_train_result(py::module_& m, const std::string& name_suffix) {
    using namespace svm;
    using result_t = train_result<Task>;

    const std::string name = "train_result" + name_suffix;

    py::class_<result_t>(m, name.c_str())
        .def(py::init())
        .DEF_ONEDAL_PROPERTY(model, result_t)
        .DEF_ONEDAL_PROPERTY(support_vectors, result_t)
        .DEF_ONEDAL_PROPERTY(support_indices, result_t)
        .DEF_ONEDAL_PROPERTY(coeffs, result_t)
        .DEF_ONEDAL_PROPERTY(biases, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m, const std::string& name_suffix) {
    using namespace svm;
    using result_t = infer_result<Task>;

    const std::string name = "infer_result" + name_suffix;

    auto cls = py::class_<result_t>(m, name.c_str())
        .def(py::init())
        .DEF_ONEDAL_PROPERTY(labels, result_t);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;
    constexpr bool is_nu_classification = std::is_same_v<Task, task::nu_classification>;

    if constexpr (is_classification || is_nu_classification) {
        cls.def_property("decision_function", &result_t::get_decision_function, &result_t::template set_decision_function<>);
    }
}

INSTANTIATOR(init_model);
INSTANTIATOR(init_train_result);
INSTANTIATOR(init_infer_result);
INSTANTIATOR(init_train_ops);
INSTANTIATOR(init_infer_ops);

TYPE2STR(svm::task::classification, "classification");
TYPE2STR(svm::task::regression, "regression");
TYPE2STR(svm::task::nu_classification, "nu_classification");
TYPE2STR(svm::task::nu_regression, "nu_regression");
TYPE2STR(dal::detail::host_policy, "");
TYPE2STR(dal::detail::data_parallel_policy, "");

void init_svm(py::module_& m) {
    using namespace svm;
    using task_list = types<task::classification, task::regression, task::nu_classification, task::nu_regression>;
    auto sub = m.def_submodule("svm");

    ONEDAL_INSTANTIATE(init_train_ops, sub, dal::detail::host_policy, task_list);
    ONEDAL_INSTANTIATE(init_train_ops, sub, dal::detail::data_parallel_policy, task_list);

    ONEDAL_INSTANTIATE(init_infer_ops, sub, dal::detail::host_policy, task_list);
    ONEDAL_INSTANTIATE(init_infer_ops, sub, dal::detail::data_parallel_policy, task_list);

    ONEDAL_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_INSTANTIATE(init_infer_result, sub, task_list);
}

} // namespace oneapi::dal
