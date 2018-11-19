/* file: algorithm_base_mode_online.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_BASE_STREAM_H__
#define __ALGORITHM_BASE_STREAM_H__

namespace daal
{
namespace algorithms
{
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm<online> is, in turn, the base class
 *        for the classes interfacing the major stages of data processing in %online mode:
 *        Analysis<online> and Training<online>.
 */
template class Algorithm<online>;
/** @} */
}
}
}
#endif
