/* file: gbt_internal.h */
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
//  Internal definitions of gradient boosted trees algorithm
//--
*/

#ifndef __GBT_INTERNAL__
#define __GBT_INTERNAL__

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{

/**
* Parallel options in GBT training algorithm
*/
enum ParallelOptions
{
    parallelFeatures = 1, /*!< Process features in parallel threads when finding splits */
    parallelNodes = 2,    /*!< Process nodes in parallel threads when building a tree */
    parallelTrees = 4,    /*!< Multi-class classification only: process single class tree in a separate thread */
    parallelAll = (parallelFeatures | parallelNodes | parallelTrees)
};

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
