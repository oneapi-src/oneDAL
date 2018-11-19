/* file: algorithm_kernel.h */
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
//  Implementation of base classes defining algorithm kernel.
//--
*/

#ifndef __ALGORITHM_KERNEL_H__
#define __ALGORITHM_KERNEL_H__

#include "services/daal_memory.h"
#include "services/daal_kernel_defines.h"
#include "services/error_handling.h"
#include "services/env_detect.h"
#include "algorithms/algorithm_types.h"

namespace daal
{
namespace algorithms
{

/**
 * @addtogroup base_algorithms
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL"></a>
 * \brief %Base class to represent algorithm implementation
 */
class Kernel
{
public:
    Kernel() {};

    virtual ~Kernel () {}
};


/** @} */
} // namespace interface1
using interface1::Kernel;

}
}
#endif
