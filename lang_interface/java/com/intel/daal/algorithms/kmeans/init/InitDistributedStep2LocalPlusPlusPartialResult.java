/* file: InitDistributedStep2LocalPlusPlusPartialResult.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2LOCALPLUSPLUSPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing initial centroids for
 *        the K-Means algorithm in the distributed processing mode
 */
public final class InitDistributedStep2LocalPlusPlusPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs empty object
     * @param context       Context to manage the partial result of computing initial centroids for the K-Means algorithm
     */
    public InitDistributedStep2LocalPlusPlusPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitDistributedStep2LocalPlusPlusPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets a partial result object for computing initial centroids for the K-Means algorithm
     * @param id   Identifier of the partial result object
     * @param val  Value of the partial result object     */
    public void set(InitDistributedStep2LocalPlusPlusPartialResultId id, NumericTable val) {
        if ((id == InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep3) ||
            (id == InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep5)) {
            cSetTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns a partial result of computing initial centroids for the K-Means algorithm
     * @param id Identifier of the partial result object
     * @return   Partial result object that corresponds to the given identifier
     */
    public NumericTable get(InitDistributedStep2LocalPlusPlusPartialResultId id) {
        if ((id == InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep3) ||
            (id == InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep5)) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets a partial result object for computing initial centroids for the K-Means algorithm
    * @param id   Identifier of the partial result object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InitDistributedStep2LocalPlusPlusPartialResultDataId id, DataCollection val) {
        if (id == InitDistributedStep2LocalPlusPlusPartialResultDataId.internalResult) {
            cSetDataCollection(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Returns a partial result object for computing initial centroids for the K-Means algorithm
    * @param id Identifier of the partial result object
    * @return   Partial result object that corresponds to the given identifier
    */
    public DataCollection get(InitDistributedStep2LocalPlusPlusPartialResultDataId id) {
        if (id != InitDistributedStep2LocalPlusPlusPartialResultDataId.internalResult) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetDataCollection(cObject, id.getValue()));
    }

    private native long cNewPartialResult();

    private native void cSetTable(long inputAddr, int id, long ntAddr);
    private native long cGetTable(long inputAddr, int id);

    private native void cSetDataCollection(long inputAddr, int id, long ntAddr);
    private native long cGetDataCollection(long inputAddr, int id);

}
/** @} */
