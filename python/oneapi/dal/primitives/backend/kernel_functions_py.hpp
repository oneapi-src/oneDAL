#pragma once

#include <pybind11/pybind11.h>

#include "oneapi/dal/common/backend/type_utils_py.hpp"

namespace oneapi::dal {

namespace py = pybind11;

template <typename Result>
void init_kernel_result(py::module_& m, const std::string& name_suffix) {
    const std::string name = "result" + name_suffix;
    py::class_<Result>(m, name.c_str())
        .def(py::init())
        .def_property("values", &Result::get_values, &Result::set_values);
}

} // namespace oneapi::dal
