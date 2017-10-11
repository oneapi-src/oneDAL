/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#ifndef _HLAPI_H_INCLUDED_
#define _HLAPI_H_INCLUDED_

#include <daal.h>
#include "cnc4daal.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace daal;

typedef daal::services::SharedPtr< std::vector< std::vector< daal::byte > > > BytesArray;

template< typename T >
inline bool use_default(const T & attr)
{
    return attr == (T)-1;
}

template< typename T >
inline bool use_default(const daal::services::SharedPtr<T> * attr)
{
    return attr->get() == NULL;
}

template< typename T >
inline bool use_default(const daal::services::SharedPtr<T> & attr)
{
    return attr.get() == NULL;
}

template<>
inline bool use_default(const std::string & attr)
{
    return attr.length() == 0;
}

template<>
inline bool use_default(const double & attr)
{
    return attr != attr;
}

template<>
inline bool use_default(const float & attr)
{
    return attr != attr;
}

inline bool string2bool(const std::string & s)
{
    if(s == "True" || s == "true" || s == "1") return true;
    if(s == "False" || s == "false" || s == "0") return false;
    throw std::invalid_argument("Bool must be one of {'True', 'true', '1', 'False', 'false', '0'}");
}

class algo_manager_i
{
public:
    virtual ~algo_manager_i() {}
};

static NTYPE as_native_shared_ptr(services::SharedPtr< const algo_manager_i > algo)
{
    int gc = 0;
    MK_DAALPTR(ret, new services::SharedPtr< const algo_manager_i >(algo), services::SharedPtr< algo_manager_i >, gc);
    TMGC(gc);
    return ret;
}

struct TableOrFList
{
    mutable daal::data_management::NumericTablePtr table;
    std::string                            file;
    std::vector< daal::data_management::NumericTablePtr > tlist;
    std::vector< std::string >             flist;
    //operator daal::data_management::NumericTablePtr() const { return table; }
    //operator bool() const { return table or flist.size() > 0; }
#ifdef _DIST_
    void serialize(CnC::serializer & ser)
    {
        ser & table & flist;
    }
#endif
};

static const daal::data_management::NumericTablePtr readCSV(const std::string& fname)
{
    data_management::FileDataSource< data_management::CSVFeatureManager >
        dataSource(fname,
                   data_management::DataSource::doAllocateNumericTable,
                   data_management::DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
}

#endif // _HLAPI_H_INCLUDED_
