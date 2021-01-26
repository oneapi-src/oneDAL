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

#include <string>

namespace oneapi::dal::preview {

class ONEDAL_EXPORT graph_csv_data_source {
public:
    graph_csv_data_source(std::string filename) : _file_name(filename) {}
    std::string get_filename() const {
        return _file_name;
    }

private:
    std::string _file_name;
};

} // namespace oneapi::dal::preview
