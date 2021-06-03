#pragma once

#include <tuple>
#include <pybind11/pybind11.h>

#define INSTANTIATOR(func_name)                                              \
struct instantiator_##func_name {                                            \
    instantiator_##func_name(py::module_& m, const std::string& name_suffix) \
        : m(m), name_suffix(name_suffix) {}                                  \
                                                                             \
    template <typename... Args>                                              \
    constexpr void run() {                                                   \
        func_name<Args...>(m, name_suffix);                                  \
    }                                                                        \
                                                                             \
    py::module_ m;                                                           \
    std::string name_suffix;                                                 \
}

#define ONEDAL_INSTANTIATE(func_name, module, ...) instantiate<instantiator_##func_name, __VA_ARGS__>(module)

#define DEF_ONEDAL_PROPERTY(name, parent) def_property(#name, &parent::get_##name, &parent::set_##name)
#define DEF_ONEDAL_PROPERTY_T(name, parent, T) def_property(#name, &parent::template get_##name<T>, &parent::template set_##name<T>)

namespace oneapi::dal {

namespace py = pybind11;

template <class Head, class... Tail>
struct types {
    using head = Head;
    using last = types<Tail...>;
    static constexpr bool has_last = true;
};

template <typename T>
struct types<T> {
    using head = T;
    using last = types<T>;
    static constexpr bool has_last = false;
};

template <typename... Args>
struct is_type_list : public std::false_type {};

template <typename... Args>
struct is_type_list<types<Args...>> : public std::true_type {};

template <typename T>
struct type_to_str;

#define TYPE2STR(type, name)      \
template <>                       \
struct type_to_str<type> {        \
    std::string operator()() {    \
        return name;              \
    }                             \
}

TYPE2STR(float, "_f32");
TYPE2STR(double, "_f64");

template <typename Index, typename T>
struct iterator {
    iterator(py::module_& m, const std::string& name_suffix = "")
        : m(m), name_suffix(name_suffix) {}

    template <typename... Args>
    void run() {
        iterate_head<Index, Args...>();
    }

    template <typename Y, typename... Args>
    void iterate_head() {
        if constexpr (is_type_list<Y>::value) {
            const auto str = type_to_str<typename Y::head>()();
            py::module_ sub = m;
            if (str != "") {
                sub = m.def_submodule(str.c_str());
            }
            T(sub, name_suffix).template run<typename Y::head, Args...>();
            if constexpr (Y::has_last) {
                iterate_head<typename Y::last, Args...>();
            }
        }
        else {
            T(m, type_to_str<Y>()() + name_suffix).template run<Y, Args...>();
        }
    }

    py::module_ m;
    std::string name_suffix;
};

template <typename Iter>
void instantiate_impl(py::module_& m) {
    Iter(m).run();
}

template <typename Iter, typename T, typename... Args>
void instantiate_impl(py::module_& m) {
    instantiate_impl<iterator<T, Iter>, Args...>(m);
}

template <typename Instantiator, typename T, typename... Args>
void instantiate(py::module_& m) {
    instantiate_impl<iterator<T, Instantiator>, Args...>(m);
}

#define ONEDAL_PARAM_DISPATCH_SECTION_START(type, name)                 \
template <typename Functor>                                             \
struct param_dispatcher_##name {                                        \
    param_dispatcher_##name(const Functor& f) : f(f) {}                 \
                                                                        \
    template <typename... Args>                                         \
    auto dispatch(const py::dict& params) {                             \
        auto value = params[#name].cast<type>();                        \
        return dispatch_impl<Args...>(value, params);                   \
    }                                                                   \
                                                                        \
    template <typename... Args>                                         \
    auto dispatch_impl(const type& value, const py::dict& params);      \
                                                                        \
    Functor f;                                                          \
};                                                                      \
                                                                        \
template <typename Functor>                                             \
template<typename... Args>                                              \
auto param_dispatcher_##name<Functor>::dispatch_impl(const type& value, \
                                            const py::dict& params)

#define ONEDAL_PARAM_DISPATCH_VALUE(value_case, result)                 \
if (value == value_case) {                                              \
    return f.template dispatch<result, Args...>(params);                \
} else                                                                  \

#define ONEDAL_PARAM_DISPATCH_SECTION_END(name)                         \
{                                                                       \
    throw std::runtime_error("Invalid value for parameter <"#name">");  \
}

#define OP_CALLER(opname)                                                                                           \
template <typename Policy, typename Input, typename Result>                                                         \
struct opname##_caller {                                                                                            \
    using task_t = typename Input::task_t;                                                                          \
    static_assert(std::is_same_v<task_t, typename Result::task_t>);                                                 \
                                                                                                                    \
    opname##_caller(const Policy& policy, const Input& input)                                                       \
        : policy(policy),                                                                                           \
          input(input) {}                                                                                           \
                                                                                                                    \
    template <typename Float, typename Method, typename... Args>                                                    \
    void call(const py::dict& params) {                                                                             \
        auto desc = get_descriptor<Float, Method, task_t, Args...>(params);                                         \
        result = dal::opname(policy, desc, input);                                                                  \
    }                                                                                                               \
                                                                                                                    \
    Policy policy;                                                                                                  \
    Input input;                                                                                                    \
    Result result;                                                                                                  \
}

} // namespace oneapi::dal
