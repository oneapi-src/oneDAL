/* file: utilities.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __SERVICES_INTERNAL_UTILITIES_H__
#define __SERVICES_INTERNAL_UTILITIES_H__

#include "services/daal_shared_ptr.h"

namespace daal
{
namespace services
{
namespace internal
{
/** @ingroup services_internal
 * @{
 */

template <typename T>
inline const T & minValue(const T & a, const T & b)
{
    return !(b < a) ? a : b;
}

template <typename T>
inline const T & maxValue(const T & a, const T & b)
{
    return (a < b) ? b : a;
}

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__IMPLACCESSOR"></a>
 *  \brief Implements method to access implementation object
 *         for PIMPL template
 */
class ImplAccessor
{
public:
    template <typename Impl, typename Class>
    static const SharedPtr<Impl> & getImplPtr(const Class & object)
    {
        return object.getImplPtr();
    }

private:
    ImplAccessor() {}
};

/** @} */

} // namespace internal
} // namespace services
} // namespace daal

#endif
