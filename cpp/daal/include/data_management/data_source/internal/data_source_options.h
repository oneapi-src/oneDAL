/* file: data_source_options.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __DATA_SOURCE_INTERNAL_DATA_SOURCE_OPTIONS__
#define __DATA_SOURCE_INTERNAL_DATA_SOURCE_OPTIONS__

namespace daal
{
namespace data_management
{
namespace internal
{
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCEOPTIONSIMPL"></a>
 *  \brief Class that helps to define data source options
 */
template <typename Value>
class DataSourceOptionsImpl
{
public:
    static Value unite(const Value & lhs, const Value & rhs) { return (Value)((int)lhs | (int)rhs); }

    explicit DataSourceOptionsImpl(Value flags) : _value(flags) {}

    bool getFlag(Value flag) const { return ((int)_value & (int)flag) != 0; }

private:
    Value _value;
};

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
