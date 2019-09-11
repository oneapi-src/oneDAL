/* file: engine_batch.h */
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
//  Implementation of engine methods.
//--
*/
#ifndef __ENGINE_BATCH__
#define __ENGINE_BATCH__

#include "engine_types.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace interface1
{

template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *algInput = static_cast<const Input *>(input);

    set(randomNumbers, algInput->get(tableToFill));
    return services::Status();
}

} // namespace interface1
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
