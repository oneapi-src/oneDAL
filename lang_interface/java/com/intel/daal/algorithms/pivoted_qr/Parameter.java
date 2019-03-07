/* file: Parameter.java */
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
 * @ingroup pivoted_qr
 * @{
 */
package com.intel.daal.algorithms.pivoted_qr;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__PARAMETER"></a>
 * @brief Pivoted QR algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of the Pivoted QR algorithm
     * @param context   Context to manage the parameter of the Pivoted QR algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets parameter of the pivoted QR algorithm
     * @param permutedColumns  On entry, if i-th element of permutedColumns != 0,
     *                          the i-th column of input matrix is moved  to the beginning of Data * P before
     *                          the computation, and fixed in place during the computation.
     *                          If i-th element of permutedColumns = 0, the i-th column of input data
     *                          is a free column (that is, it may be interchanged during the
     *                          computation with any other free column).
     */
    public void setPermutedColumns(NumericTable permutedColumns) {
        cSetPermutedColumns(this.cObject, permutedColumns.getCObject());
    }

    /**
     * Gets parameter of  the pivoted QR algorithm
     * @return    Identifier of the parameter
     */
    public NumericTable getPermutedColumns() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPermutedColumns(this.cObject));
    }

    private native void cSetPermutedColumns(long parAddr, long permutedColumnsAddr);

    private native long cGetPermutedColumns(long parAddr);
}
/** @} */
