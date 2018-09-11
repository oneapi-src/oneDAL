/* file: service_algo_utils.h */
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
//  Declaration of service utilities used with services structures
//--
*/
#ifndef __SERVICE_ALGO_UTILS_H__
#define __SERVICE_ALGO_UTILS_H__

#include "services/host_app.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
    class AlgorithmIface;
    class Input;
}
}

namespace services
{
namespace internal
{

services::HostAppIface* hostApp(algorithms::interface1::Input& inp);
void setHostApp(const services::SharedPtr<services::HostAppIface>& pHostApp, algorithms::interface1::Input& inp);
services::HostAppIfacePtr getHostApp(daal::algorithms::interface1::Input& inp);
bool isCancelled(services::Status& s, services::HostAppIface* pHostApp);

//////////////////////////////////////////////////////////////////////////////////////////
// Helper class handling cancellation status depending on the number of jobs to be done
//////////////////////////////////////////////////////////////////////////////////////////
class HostAppHelper
{
public:
    HostAppHelper(HostAppIface* hostApp, size_t maxJobsBeforeCheck);
    bool isCancelled(services::Status& s, size_t nJobsToDo);
    void setup(size_t maxJobsBeforeCheck);
    void reset(size_t maxJobsBeforeCheck);

private:
    services::HostAppIface* _hostApp;
    size_t _maxJobsBeforeCheck; //granularity
    size_t _nJobsAfterLastCheck;
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
