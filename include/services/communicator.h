/* file: communicator.h */
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

#ifndef __DAAL_SERVICES_COMMUNICATOR_H__
#define __DAAL_SERVICES_COMMUNICATOR_H__

#include "services/daal_shared_ptr.h"
#include "services/internal/utilities.h"
#include "oneapi/internal/communicator.h"

namespace daal
{
namespace preview
{
namespace services
{
namespace interface1
{
/**
 * @defgroup communicator COMMUNICATOR*
 * \brief Contains classes designed to work with communication layer and call
 * oneAPI implementations of algorithms
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__COMMUNICATOR"></a>
 *  \brief Base class for device information needed to perform
 *   computations
 */
class Communicator : public Base
{
    friend class daal::services::internal::ImplAccessor;

private:
    typedef daal::preview::comm::internal::interface1::CommunicatorIface ImplType;

public:
    Communicator() {}

protected:
    explicit Communicator(ImplType * impl) : _impl(impl) {}

    const daal::services::SharedPtr<ImplType> & getImplPtr() const { return _impl; }

private:
    daal::services::SharedPtr<ImplType> _impl;
};
/**
 *  <a name="DAAL-CLASS-SERVICES__EMPTYCOMMUNICATOR"></a>
 *  \brief Stub for disabled inter-process communication
  */

class EmptyCommunicator : public Communicator
{
public:
    EmptyCommunicator() : Communicator(new daal::preview::comm::internal::DummyCommunicator()) {}
};

/** @} */
} // namespace interface1
using interface1::Communicator;
} // namespace services
} // namespace preview
} // namespace daal

#ifdef DAAL_SYCL_INTERFACE
    #include "oneapi/internal/communicator_oneccl_sycl.h"

namespace daal
{
namespace preview
{
namespace services
{
namespace interface1
{
class OneCclCommunicator : public Communicator
{
public:
    OneCclCommunicator(cl::sycl::queue & deviceQueue) : Communicator(createCommunicator(deviceQueue)) {}

private:
    static daal::preview::comm::internal::CommunicatorIface * createCommunicator(cl::sycl::queue & queue)
    {
        return new daal::preview::comm::internal::CommunicatorOneCclImpl(queue);
    }
};
/** @} */
} // namespace interface1

using interface1::OneCclCommunicator;

} // namespace services
} // namespace preview
} // namespace daal
#endif // DAAL_SYCL_INTERFACE

#endif
