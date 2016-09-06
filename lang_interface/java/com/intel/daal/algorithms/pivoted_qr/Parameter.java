/* file: Parameter.java */
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

package com.intel.daal.algorithms.pivoted_qr;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__PARAMETER"></a>
 * @brief Pivoted QR algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

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
    public HomogenNumericTable getPermutedColumns() {
        return new HomogenNumericTable(getContext(), cGetPermutedColumns(this.cObject));
    }

    private native void cSetPermutedColumns(long parAddr, long permutedColumnsAddr);

    private native long cGetPermutedColumns(long parAddr);
}
