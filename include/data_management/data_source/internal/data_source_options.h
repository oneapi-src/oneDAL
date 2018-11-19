/* file: data_source_options.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
template<typename Value>
class DataSourceOptionsImpl
{
public:
    static Value unite(const Value &lhs, const Value &rhs)
    {
        return (Value)( (int)lhs | (int)rhs );
    }

    explicit DataSourceOptionsImpl(Value flags) :
        _value(flags) { }

    bool getFlag(Value flag) const
    {
        return ((int)_value & (int)flag) != 0;
    }

private:
    Value _value;
};


} // namespace internal
} // namespace data_management
} // namespace daal

#endif
