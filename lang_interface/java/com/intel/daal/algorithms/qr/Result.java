/* file: Result.java */
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
 * @ingroup qr_without_pivoting
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__RESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of QR decomposition algorithm
 *        in the batch processing mode or finalizeCompute() method in the online processing mode  for the algorithm on the second or third
 *        steps in the distributed processing mode
 */
public class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the QR decomposition algorithm
     * @param context   Context to manage the result of the QR decomposition algorithm
     */
    public Result(DaalContext context) {
        super(context);
        cObject = cNewResult();
    }

    public Result(DaalContext context, long cResult) {
        super(context);
        this.cObject = cResult;
    }

    /**
     * Returns the result of the QR decomposition algorithm
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.matrixQ && id != ResultId.matrixR) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets the result of the QR decomposition algorithm
     * @param id    Identifier of the result
     * @param value NumericTable to store result
     */
    public void set(ResultId id, NumericTable value) {
        if (id != ResultId.matrixQ && id != ResultId.matrixR) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cResult, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
/** @} */
