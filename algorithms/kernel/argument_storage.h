/* file: argument_storage.h */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation
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

/*
//++
//  Declaration of internal argument storage class
//--
*/
#ifndef __ARGUMENT_STORAGE_H__
#define __ARGUMENT_STORAGE_H__
#include "data_collection.h"
#include "service_defines.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
class ArgumentStorage : public data_management::DataCollection
{
public:
    enum Extension
    {
        hostApp = 0
    };
    DAAL_CAST_OPERATOR(ArgumentStorage);
    ArgumentStorage(const size_t n) : data_management::DataCollection(n) {}
    ArgumentStorage(const ArgumentStorage & o) : data_management::DataCollection(o), _extensions(o._extensions) {}
    virtual ~ArgumentStorage() {}

    services::SharedPtr<Base> getExtension(Extension type);
    void setExtension(Extension type, const services::SharedPtr<Base> & ptr);

protected:
    typedef services::SharedPtr<Base> BasePtr;
    typedef services::Collection<BasePtr> BasePtrCollection;
    BasePtrCollection _extensions;
};

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
