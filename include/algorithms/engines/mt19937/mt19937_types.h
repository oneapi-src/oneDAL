/* file: mt19937_types.h */
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
//  Implementation of mt19937 engine.
//--
*/

#ifndef __MT19937_TYPES_H__
#define __MT19937_TYPES_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
/**
 * @defgroup engines_mt19937 Mt19937 Engine
 * \copydoc daal::algorithms::engines::mt19937
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mt19937 engine
 */
namespace mt19937
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MT19937__METHOD"></a>
 * Available methods to compute mt19937 engine
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

} // namespace mt19937
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
