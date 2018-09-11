/* file: mcg59_types.h */
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
//  Implementation of mcg59 engine.
//--
*/

#ifndef __MCG59_TYPES_H__
#define __MCG59_TYPES_H__

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
 * @defgroup engines_mcg59 Mcg59 Engine
 * \copydoc daal::algorithms::engines::mcg59
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mcg59 engine
 */
namespace mcg59
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MCG59__METHOD"></a>
 * Available methods to compute mcg59 engine
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

} // namespace mcg59
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
