/* file: OptionalResultId.java */
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
 * @ingroup iterative_solver
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALRESULTID"></a>
 * @brief Available result identifiers for the iterative solver algorithm
 */
public final class OptionalResultId {
    private int _value;

    /**
     * Constructs the optional result object identifier using the provided value
     * @param value     Value corresponding to the optional result object identifier
     */
    public OptionalResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional result object identifier
     * @return Value corresponding to the optional result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int optionalResultId = 2;

    public static final OptionalResultId optionalResult  = new OptionalResultId(optionalResultId); /*!< Algorithm-specific result data */
}
/** @} */
