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

#include "oneapi/dal/io/detail/load_graph_service.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "src/externals/service_service.h"
#include "src/threading/threading.h"

using namespace std;
namespace oneapi::dal::preview::load_graph::detail {

ONEAPI_DAL_EXPORT int daal_string_to_int(const char* nptr, char** endptr) {
    return daal::internal::Service<>::serv_string_to_int(nptr, endptr);
}
} // namespace oneapi::dal::preview::load_graph::detail

ONEAPI_DAL_EXPORT void _daal_threader_for_oneapi(int n,
                                                 int threads_request,
                                                 const void* a,
                                                 oneapi::dal::preview::functype func) {
    _daal_threader_for(n, threads_request, a, static_cast<daal::functype>(func));
}
