/* file: mt2203_types.h */
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
//  Implementation of mt2203 engine.
//--
*/

#ifndef __MT2203_TYPES_H__
#define __MT2203_TYPES_H__

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
 * @defgroup engines_mt2203 Mt2203 Engine
 * \copydoc daal::algorithms::engines::mt2203
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mt2203 engine
 */
namespace mt2203
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MT2203__METHOD"></a>
 * Available methods to compute mt2203 engine
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

} // namespace mt2203
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
