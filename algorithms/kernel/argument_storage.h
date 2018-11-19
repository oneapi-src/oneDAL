/* file: argument_storage.h */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
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
    ArgumentStorage(const ArgumentStorage& o) : data_management::DataCollection(o), _extensions(o._extensions){}
    virtual ~ArgumentStorage() {}

    services::SharedPtr<Base> getExtension(Extension type);
    void setExtension(Extension type, const services::SharedPtr<Base>& ptr);

protected:
    typedef services::SharedPtr<Base> BasePtr;
    typedef services::Collection<BasePtr> BasePtrCollection;
    BasePtrCollection _extensions;
};

}// namespace internal
}// namespace algorithms
}// namespace daal

#endif
