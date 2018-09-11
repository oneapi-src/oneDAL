/* file: base.h */
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
//  Implementation of a dummy base class needed to fix ABI inconsistency between
//  Visual Studio* 2012 and 2013.
//--
*/

#ifndef __BASE_H__
#define __BASE_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"

namespace daal
{
/**
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-CLASS-__BASE"></a>
 * \brief %Base class for Intel(R) Data Analytics Acceleration Library objects
 */
class DAAL_EXPORT Base
{
public:
    DAAL_NEW_DELETE();
    virtual ~Base() {}
};
/** @} */
}
#endif
