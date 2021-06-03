#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/common/backend/table_conversion_py.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace oneapi::dal {

void init_table(py::module_& m) {
    py::class_<table>(m, "table")
        .def(py::init())
        .def_property_readonly("has_data", &table::has_data)
        .def_property_readonly("column_count", &table::get_column_count)
        .def_property_readonly("row_count", &table::get_row_count)
        .def_property_readonly("kind",[](const table& t) {
            if(t.get_kind() == 0) { // TODO: expose empty table kind
                return "empty";
            }
            if(t.get_kind() == homogen_table::kind()) {
                return "homogen";
            }
            return "unknown";
        });

    m.def("from_numpy", [](py::object obj){
        auto* obj_ptr = obj.ptr();
        return python::convert_to_table(obj_ptr);
    });

    m.def("to_numpy", [](dal::table& t) -> py::handle {
        auto* obj_ptr = python::convert_to_numpy(t);
        return obj_ptr;
    });
}

} // namespace oneapi::dal
