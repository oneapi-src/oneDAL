/** file service_algo_utils.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//
//--
*/

#include "error_indexes.h"
#include "error_handling.h"
#include "service_algo_utils.h"

namespace daal
{
namespace services
{
namespace interface1
{

HostAppIface::HostAppIface() :_impl(nullptr)
{
}

HostAppIface::~HostAppIface()
{
    delete _impl;
}
}// namespace interface1

namespace internal
{

bool isCancelled(services::Status& s, services::HostAppIface* pHostApp)
{
    if(!pHostApp || !pHostApp->isCancelled())
        return false;
    s.add(services::ErrorUserCancelled);
    return true;
}

HostAppHelper::HostAppHelper(HostAppIface* hostApp, size_t maxJobsBeforeCheck) :
_hostApp(hostApp), _maxJobsBeforeCheck(maxJobsBeforeCheck),
_nJobsAfterLastCheck(0)
{
}

bool HostAppHelper::isCancelled(services::Status& s, size_t nJobsToDo)
{
    if(!_hostApp)
        return false;
    _nJobsAfterLastCheck += nJobsToDo;
    if(_nJobsAfterLastCheck < _maxJobsBeforeCheck)
        return false;
    _nJobsAfterLastCheck = 0;
    return services::internal::isCancelled(s, _hostApp);
}

void HostAppHelper::setup(size_t maxJobsBeforeCheck)
{
    _maxJobsBeforeCheck = maxJobsBeforeCheck;
}

void HostAppHelper::reset(size_t maxJobsBeforeCheck)
{
    setup(maxJobsBeforeCheck);
    _nJobsAfterLastCheck = 0;
}

}// namespace internal
}// namespace services
}// namespace daal
