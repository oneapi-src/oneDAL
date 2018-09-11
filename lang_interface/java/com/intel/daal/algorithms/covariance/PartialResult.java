/* file: PartialResult.java */
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the correlation
 *        or variance-covariance matrix algorithm in the online or distributed processing mode
 */
public class PartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs an empty PartialResult object
     * @param context  Context to manage the correlation or variance-covariance matrix algorithm
    */
    public PartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public PartialResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns the partial result of the correlation or variance-covariance matrix algorithm
     * @param  id   Identifier of the partial result, @ref PartialResultId
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialResultId id) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.nObservations.getValue() && idValue != PartialResultId.crossProduct.getValue()
                && idValue != PartialResultId.sum.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialResultTable(getCObject(), idValue));
    }

    /**
     * Sets the partial result of the correlation or variance-covariance matrix algorithm
     * @param id                 Identifier of the partial result
     * @param value              Object to store the partial result
     */
    public void set(PartialResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.nObservations.getValue() && idValue != PartialResultId.crossProduct.getValue()
                && idValue != PartialResultId.sum.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResultTable(long cPartialResult, int id);

    private native void cSetPartialResultTable(long cPartialResult, int id, long cNumericTable);
}
/** @} */
