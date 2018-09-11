/* file: utilities.h */
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

#ifndef __SERVICES_INTERNAL_UTILITIES_H__
#define __SERVICES_INTERNAL_UTILITIES_H__

namespace daal
{
namespace services
{
namespace internal
{

template <typename T>
inline const T & minValue(const T & a, const T & b) { return !(b < a) ? a : b; }

template <typename T>
inline const T & maxValue(const T & a, const T & b) { return (a < b) ? b : a; }

} // namespace internal
} // namespace services
} // namespace daal

#endif
