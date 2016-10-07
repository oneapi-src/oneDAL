/* file: training.h */
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

#ifndef __TRAINING_H__
#define __TRAINING_H__

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__TRAININGCONTAINERIFACE"></a>
 *  \brief Abstract interface class that provides virtual methods
 *         to access and run implementations of the model training algorithms.
 *         The class is associated with the Training class and supports the methods for computation
 *         and finalization of the training output in the batch, distributed, and online modes.
 *         The methods of the container are defined in derivative containers defined for each training algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class TrainingContainerIface : public AlgorithmContainerIface<mode> {};
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__TRAINING"></a>
 *  \brief Provides methods to train models that depend on the data provided. For example, these methods enable training the linear regression model.
 *         The methods of the class support different computation modes: batch, distributed, and online(see \ref ComputeMode).
 *         Classes that implement specific algorithms of model training are derived classes of the Training class.
 *         The class additionally provides methods for validation of input and output parameters
 *         of the algorithms.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class Training               : public AlgorithmImpl<mode>     {};
/** @} */
}
}
#endif
