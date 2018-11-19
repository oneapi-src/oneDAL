/* file: training.h */
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

#ifndef __TRAINING_H__
#define __TRAINING_H__

namespace daal
{
namespace algorithms
{
/**
 * @defgroup training_and_prediction Training and Prediction
 * \brief Contains classes of machine learning algorithms.
 *        Unlike analysis algorithms, which are intended to characterize the structure of data sets,
 *        machine learning algorithms model the data.
 *        Modeling operates in two major stages: training and prediction or decision making.
 * @ingroup algorithms
 */
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
template<ComputeMode mode> class TrainingContainerIface : public AlgorithmContainerImpl<mode> {};
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
