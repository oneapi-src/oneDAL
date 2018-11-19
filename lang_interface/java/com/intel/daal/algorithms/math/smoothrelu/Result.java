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
 * @ingroup smoothrelu
 * @{
 */
package com.intel.daal.algorithms.math.smoothrelu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SMOOTHRELU__RESULT"></a>
 * @brief Results obtained with the compute() method of the SmoothReLU algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the SmoothReLU algorithm
     * @param context   Context to manage the result of the SmoothReLU algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of SmoothReLU algorithm
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        int idValue = id.getValue();
        if (idValue != ResultId.value.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetValue(cObject));
    }

    /**
     * Sets the final result of the SmoothReLU algorithm
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(ResultId id, NumericTable val) {
        int idValue = id.getValue();
        if (idValue != ResultId.value.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetValue(cObject, val.getCObject());
    }

    private native long cNewResult();

    private native long cGetValue(long cObject);

    private native void cSetValue(long cObject, long cNumericTable);
}
/** @} */
