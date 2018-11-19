/* file: analysis.h */
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

#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include "algorithms/algorithm_base.h"

namespace daal
{
/**
 * @defgroup algorithms Algorithms
 * @{
 */
namespace algorithms
{
/**
 * @defgroup analysis Analysis
 * \brief Contains classes for analysis algorithms that are intended to uncover the underlying structure
 *        of a data set and to characterize it by a set of quantitative measures, such as statistical moments,
 *        correlations coefficients, and so on.
 * @ingroup algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISCONTAINERIFACE"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the analysis algorithms. It is associated with the Analysis class
 *        and supports the methods for computation and finalization of the analysis results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers
 *        defined for each algorithm of data analysis.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class AnalysisContainerIface : public AlgorithmContainerImpl<mode>
{
public:
    AnalysisContainerIface(daal::services::Environment::env *daalEnv = 0): AlgorithmContainerImpl<mode>(daalEnv) {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSIS"></a>
 * \brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates.
 *        The methods of the class support different computation modes: batch, distributed, and online(see \ref ComputeMode).
 *        Classes that implement specific algorithms of the data analysis are derived classes of the \ref Analysis class.
 *        The class additionally provides virtual methods for validation of input and output parameters
 *        of the algorithms.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class Analysis: public AlgorithmImpl<mode> {};
}
}
/** @} */
/** @} */
#endif
