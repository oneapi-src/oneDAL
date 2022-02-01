/* file: numeric_types.h */
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

} // namespace data_management
} // namespace daal
#endif
