/* file: input_collection.h */
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

#ifndef __INPUT_COLLECTION_H__
#define __INPUT_COLLECTION_H__

#include "algorithms/algorithm_types.h"
#include "data_management/data/data_collection.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
typedef KeyValueCollection<algorithms::Input> KeyValueInputCollection;
typedef services::SharedPtr<KeyValueInputCollection> KeyValueInputCollectionPtr;
typedef services::SharedPtr<const KeyValueInputCollection> KeyValueInputCollectionConstPtr;

/** @} */
} // namespace interface1
using interface1::KeyValueInputCollection;
using interface1::KeyValueInputCollectionPtr;
using interface1::KeyValueInputCollectionConstPtr;

} // namespace data_management
} // namespace daal

#endif
