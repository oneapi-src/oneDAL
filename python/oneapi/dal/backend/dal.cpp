#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace oneapi::dal {

void init_table(py::module_&);
void init_policy(py::module_& m);

void init_linear_kernel(py::module_&);
void init_rbf_kernel(py::module_&);
void init_polynomial_kernel(py::module_&);

void init_svm(py::module_& m);

PYBIND11_MODULE(_onedal_py_dpc, m) {
    init_policy(m);
    init_table(m);

    init_linear_kernel(m);
    init_rbf_kernel(m);
    init_polynomial_kernel(m);

    init_svm(m);
}

} // namespace oneapi::dal
