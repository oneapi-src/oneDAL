/* file: prediction.h */
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
class PredictionContainerIface : public AlgorithmContainerImpl<batch> {};

class DistributedPredictionContainerIface : public AlgorithmContainerImpl<distributed> {};
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PREDICTION"></a>
 *  \brief Provides prediction methods depending on the model such as linear_regression::Model.
 *         The methods of the class support different computation modes: batch, distributed, and online(see \ref ComputeMode).
 *         Classes that implement specific algorithms of the model based data prediction are derived classes of the Prediction class.
 *         The class additionally provides virtual methods for validation of input and output parameters of the algorithms.
 */
class Prediction               : public AlgorithmImpl<batch>       {};

class DistributedPrediction    : public AlgorithmImpl<distributed> {};
/** @} */
}
}
#endif
