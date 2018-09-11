/* file: InitPartialResult.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing initial centroids for
 *        the K-Means algorithm in the distributed processing mode
 */
public class InitPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs empty InitPartialResult
     * @param context       Context to manage the partial result of computing initial centroids for the K-Means algorithm
     */
    public InitPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of computing initial centroids for the K-Means algorithm
     * @param  id   Identifier of InitPartialResult, @ref PartialResultId
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(InitPartialResultId id) {
        int idValue = id.getValue();
        if (idValue != InitPartialResultId.partialClustersNumber.getValue()
                && idValue != InitPartialResultId.partialCentroids.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        long tbl = cGetPartialResultTable(getCObject(), idValue);
        if(tbl == 0)
            return null;
        return (NumericTable)Factory.instance().createObject(getContext(), tbl);
    }

    /**
     * Sets a partial result of computing initial centroids for the K-Means algorithm
     * @param id                 Identifier of the partial result
     * @param value              Value of the partial result
     */
    public void set(InitPartialResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (idValue != InitPartialResultId.partialClustersNumber.getValue()
                && idValue != InitPartialResultId.partialCentroids.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResultTable(long cPartialResult, int id);

    private native void cSetPartialResultTable(long cPartialResult, int id, long cNumericTable);
}
/** @} */
