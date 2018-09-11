/* file: host_app.h */
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

/*
//++
//  Interface of host application class used by the library
//--
*/

#ifndef __DAAL_HOST_APP_H__
#define __DAAL_HOST_APP_H__

#include "services/daal_defines.h"
#include "services/base.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace services
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 *  <a name="DAAL-CLASS-SERVICES__HOSTAPPIFACE"></a>
 *  \brief Abstract class which defines callback interface for the host application of this library
 *         to enable such features as computation cancelling, progress bar, status bar, verbose, etc.
 */
class DAAL_EXPORT HostAppIface : public Base
{
public:
    DAAL_NEW_DELETE();
    HostAppIface();
    virtual ~HostAppIface();
    /**
     * This callback is called by compute() methods of the library algorithms.
     * If it returns true then compute() stops and returns 'ErrorUserCancelled' status
     * \return True when algorithm should be aborted
     */
    virtual bool isCancelled() = 0;

private:
    Base* _impl;
};
typedef services::SharedPtr<HostAppIface> HostAppIfacePtr;

} // namespace interface1
using interface1::HostAppIface;
using interface1::HostAppIfacePtr;

}
}
#endif //__DAAL_HOST_APP_H__
