/* file: comm_detect.h */
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

/*
//++
//  Implementation of the class used for communication layer.
//--
*/

#ifndef __COMM_DETECT_H__
#define __COMM_DETECT_H__

#include "services/base.h"
#include "services/daal_defines.h"
#include "services/communicator.h"

namespace daal
{
namespace preview
{
namespace services
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-SERVICES__COMMMANAGER"></a>
 * \brief Class that provides methods to interact with the environment, including processor detection and control by the number of threads
 */
class DAAL_EXPORT CommManager : public Base
{
public:
    static CommManager * getInstance();

    void setDefaultCommunicator(const preview::services::interface1::Communicator & comm)
    {
        _communicator = daal::services::internal::ImplAccessor::getImplPtr<daal::preview::comm::internal::CommunicatorIface>(comm);
    }
    daal::preview::comm::internal::CommunicatorIface & getDefaultCommunicator() { return *_communicator; }

private:
    CommManager() { this->setDefaultCommunicator(EmptyCommunicator()); }
    CommManager(const CommManager & e);
    CommManager & operator=(const CommManager &);
    ~CommManager() {}

    daal::services::SharedPtr<daal::preview::comm::internal::CommunicatorIface> _communicator;
};
} // namespace interface1

using interface1::CommManager;

} // namespace services
} // namespace preview
/** @} */
} // namespace daal
#endif
