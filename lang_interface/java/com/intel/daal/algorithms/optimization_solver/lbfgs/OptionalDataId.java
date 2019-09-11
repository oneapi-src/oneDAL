/* file: OptionalDataId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/**
 * @ingroup lbfgs
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.lbfgs;

/**
 * <a name="DAAL-CLASS-ALGORITHM__OPTIMIZATION_SOLVER__LBFGS__OPTIONALDATAID"></a>
 * @brief Available identifiers of input objects for the iterative algorithm
 */
public final class OptionalDataId {
    private int _value;

    /**
     * Constructs the optional data object identifier using the provided value
     * @param value     Value corresponding to the optional data object identifier
     */
    public OptionalDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional data object identifier
     * @return Value corresponding to the optional data object identifier
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
/** @} */
