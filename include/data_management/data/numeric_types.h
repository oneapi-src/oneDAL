/* file: numeric_types.h */
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
//  Declaration for types in data_management.
//--
*/


#ifndef __NUMERIC_TYPES_H__
#define __NUMERIC_TYPES_H__

namespace daal
{
namespace data_management
{
/**
 * @defgroup numeric_tables Numeric Tables
 * \brief Contains classes for a data management component responsible for representation of data in the numeric format
 * @ingroup data_management
 * @{
 */
enum ReadWriteMode
{
    readOnly  = 1,
    writeOnly = 2,
    readWrite = 3
};
/** @} */

}
} // namespace daal
#endif
