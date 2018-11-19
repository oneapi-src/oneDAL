/* file: PartialCorrelationResult.java */
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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALCORRELATIONRESULT"></a>
 * @brief Provides methods to access partial results obtained with compute() of the %correlation method of PCA algorithm
 *        in the online or distributed processing mode
 */
public final class PartialCorrelationResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result of the algorithm
     * @param context       Context to manage the partial result of the algorithm
     */
    public PartialCorrelationResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialCorrelationResult();
    }

    public PartialCorrelationResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns partial result of the PCA algorithm
     * @param  id   Identifier of the PartialCorrelationResult, PartialCorrelationResultID
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialCorrelationResultID id) {
        int idValue = id.getValue();
        if (id != PartialCorrelationResultID.nObservations && id != PartialCorrelationResultID.crossProductCorrelation
                && id != PartialCorrelationResultID.sumCorrelation) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialCorrelationResultTable(getCObject(), idValue));
    }

    /**
     * Sets partial result of the PCA algorithm
     * @param id                 Identifier of the partial result
     * @param value              Value of the partial result
     */
    public void set(PartialCorrelationResultID id, NumericTable value) {
        int idValue = id.getValue();
        if (id != PartialCorrelationResultID.nObservations && id != PartialCorrelationResultID.crossProductCorrelation
                && id != PartialCorrelationResultID.sumCorrelation) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialCorrelationResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialCorrelationResult();

    private native long cGetPartialCorrelationResultTable(long cPartialCorrelationResult, int id);

    private native void cSetPartialCorrelationResultTable(long cPartialCorrelationResult, int id, long cNumericTable);
}
/** @} */
