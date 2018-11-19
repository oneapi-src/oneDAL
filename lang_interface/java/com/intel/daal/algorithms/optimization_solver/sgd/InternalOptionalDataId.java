/* file: InternalOptionalDataId.java */
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

/**
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;
import com.intel.daal.algorithms.optimization_solver.sgd.OptionalDataId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__INTERNALOPTIONALDATAID"></a>
 * @brief Available identifiers of InternalOptionalDataId objects for the algorithm
 */
public final class InternalOptionalDataId {
    private int _value;

    /**
     * Constructs the internal optional data object identifier using the provided value
     * @param value     Value corresponding to the internal optional data object identifier
     */
    public InternalOptionalDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the internal optional data object identifier
     * @return Value corresponding to the internal optional data object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int rngStateId = OptionalDataId.pastUpdateVector.getValue() + 1;

    public static final InternalOptionalDataId rngState = new InternalOptionalDataId(rngStateId); /*!< Memory block with random numbers generator state */
}
/** @} */
