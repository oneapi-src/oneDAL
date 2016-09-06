/* file: algorithm_base_distributed.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_BASE_DISTRIBUTED_H__
#define __ALGORITHM_BASE_DISTRIBUTED_H__

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
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIFACE"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %distributed mode. It is associated with the Algorithm<distributed> class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in distributed mode.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template class AlgorithmContainerIface<distributed>;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm<distributed> is, in turn, the base class
 *        for the classes interfacing the major stages of data processing in %distributed mode:
 *        Analysis<distributed> and Training<distributed>.
 */
template class Algorithm<distributed>;
/** @} */
}
}
}
#endif
