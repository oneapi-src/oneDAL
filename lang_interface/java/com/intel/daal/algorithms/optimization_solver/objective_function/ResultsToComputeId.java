/* file: ResultsToComputeId.java */
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

package com.intel.daal.algorithms.optimization_solver.objective_function;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTSTOCOMPUTEID"></a>
 * @brief Available computation flag identifiers for the objective funtion result
 */
public final class ResultsToComputeId {

    public static final long gradient = 0x0000000000000001L; /*!< Objective function gradient compute flag */
    public static final long value    = 0x0000000000000002L; /*!< Objective function value compute flag */
    public static final long hessian  = 0x0000000000000004L; /*!< Objective function hessian compute flag */
}
