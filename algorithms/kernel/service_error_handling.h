/* file: service_error_handling.h */
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
//  Declaration of service error handling classes
//--
*/
#ifndef __SERVICE_ERROR_HANDLING_H__
#define __SERVICE_ERROR_HANDLING_H__

#include "error_handling.h"
#include "service_threading.h"

namespace daal
{

//Thread safe holder of Status
class SafeStatus
{
public:
    explicit SafeStatus();
    explicit SafeStatus(const services::Status& s);

    bool ok() const;
    operator bool() const { return ok(); }

    SafeStatus& add(services::ErrorID id);
    SafeStatus& add(const services::ErrorPtr& e);
    SafeStatus& add(const services::Status& s);
    SafeStatus& operator |=(const services::Status& other) { return add(other); }

    //Return services::Status and clean this class
    services::Status detach();

private:
    services::Status _val;
    mutable Mutex _m;
};

} // namespace daal

#endif
