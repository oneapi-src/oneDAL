/* file: prediction.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __PREDICTION_H__
#define __PREDICTION_H__

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PREDICTIONCONTAINERIFACE"></a>
 *  \brief Abstract interface class that provides virtual methods to access and run implementations
 *         of the algorithms for model based prediction. Is associated with the Prediction class
 *         and supports the methods for computing the prediction results based on the trained model.
 *         The methods of the container are defined in derivative containers defined for each prediction algorithm.
 */
class PredictionContainerIface : public AlgorithmContainerImpl<batch>
{};

class DistributedPredictionContainerIface : public AlgorithmContainerImpl<distributed>
{};
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PREDICTION"></a>
 *  \brief Provides prediction methods depending on the model such as linear_regression::Model.
 *         The methods of the class support different computation modes: batch, distributed, and online(see \ref ComputeMode).
 *         Classes that implement specific algorithms of the model based data prediction are derived classes of the Prediction class.
 *         The class additionally provides virtual methods for validation of input and output parameters of the algorithms.
 */
class Prediction : public AlgorithmImpl<batch>
{};

class DistributedPrediction : public AlgorithmImpl<distributed>
{};
/** @} */
} // namespace algorithms
} // namespace daal
#endif
