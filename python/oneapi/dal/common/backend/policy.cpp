#include "oneapi/dal/detail/policy.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace oneapi::dal {

void init_policy(py::module_& m) {
    py::class_<detail::host_policy>(m, "host_policy")
        .def(py::init());

#ifdef ONEDAL_DATA_PARALLEL
    py::class_<detail::data_parallel_policy>(m, "data_parallel_policy")
        .def(py::init([](const std::string& device_type) {
            if (device_type == "gpu") {
                return new detail::data_parallel_policy(sycl::gpu_selector());
            }

            return new detail::data_parallel_policy(sycl::cpu_selector());
        }));
#endif
}

} // namespace oneapi::dal
