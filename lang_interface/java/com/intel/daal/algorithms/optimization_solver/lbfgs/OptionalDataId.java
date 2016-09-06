/* file: OptionalDataId.java */
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

package com.intel.daal.algorithms.optimization_solver.lbfgs;

/**
 * <a name="DAAL-CLASS-ALGORITHM__OPTIMIZATION_SOLVER__LBFGS__OPTIONALDATAID"></a>
 * @brief Available identifiers of input objects for the iterative algorithm
 */
public final class OptionalDataId {
    private int _value;

    /**
     * Constructs the optional data identifier for iterative algorithm
     * @param value Value of identifier
     */
    public OptionalDataId(int value) {
        _value = value;
    }

    /**
    * Returns the value corresponding to the identifier of optional data object
    * @return Value corresponding to the identifier
    */
    public int getValue() {
        return _value;
    }

    private static final int correctionPairsId = 0;
    private static final int correctionIndicesId = 1;
    private static final int averageArgumentLIterationsId = 2;

    /*!< Correction pairs table. Numeric table 2*m x n, where
         rows (0, m-1) represent correction vectors S and rows (m, 2*m-1) represent correction vectors Y */
    public static final OptionalDataId correctionPairs = new OptionalDataId(correctionPairsId);
    /*!< Numeric table of size 1 x 2 with 32-bit integer indexes.
         The first value is the index of correction pair t,
         the second value is the index of the last iteration k from the previous run */
    public static final OptionalDataId correctionIndices = new OptionalDataId(correctionIndicesId);
    /*!< Numeric table of size 2 x n, where
         row 0 represent average arguments for the previous L iterations and
         row 1 represent average arguments for the last L iterations.
         These values are required to compute S correction vectors on the next step */
    public static final OptionalDataId averageArgumentLIterations = new OptionalDataId(averageArgumentLIterationsId);
}
