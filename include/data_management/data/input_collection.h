/* file: input_collection.h */
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

}
}

#endif
