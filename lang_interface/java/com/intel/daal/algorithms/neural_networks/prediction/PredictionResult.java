/* file: PredictionResult.java */
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
 * @ingroup neural_networks_prediction
 * @{
 */
package com.intel.daal.algorithms.neural_networks.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Provides methods to access result obtained with the compute() method of the neural networks prediction algorithm
 */
public class PredictionResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the neural networks prediction algorithm
     * @param context   Context to manage the result of the neural networks prediction algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
    }

    public PredictionResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the neural networks model based prediction
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(PredictionResultId id) {
        if (id == PredictionResultId.prediction) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of the neural networks model based prediction
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(PredictionResultCollectionId id) {
        if (id == PredictionResultCollectionId.predictionCollection) {
            return new KeyValueDataCollection(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of the neural networks model based prediction
     * @param  id    Identifier of the result
     * @param  key   Key to use to retrieve data
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(PredictionResultCollectionId id, int key) {
        if (id == PredictionResultCollectionId.predictionCollection) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetTensor(this.cObject, id.getValue(), key));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of neural networks model based prediction
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(PredictionResultId id, Tensor val) {
        if (id == PredictionResultId.prediction) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of neural networks model based prediction
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(PredictionResultCollectionId id, Tensor val) {
        if (id == PredictionResultCollectionId.predictionCollection) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of neural networks model based prediction
     * @param id    Identifier of the input object
     * @param key   Key to use to retrieve data
     * @param val   Value of the input object
     */
    public void add(PredictionResultCollectionId id, int key, Tensor val) {
        if (id == PredictionResultCollectionId.predictionCollection) {
            cAddTensor(this.cObject, id.getValue(), key, val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }
    }

    private native long cGetValue(long cObject, int id);
    private native void cSetValue(long cObject, int id, long ntAddr);

    private native void cAddTensor(long cObject, int id, int key, long ntAddr);
    private native long cGetTensor(long cObject, int id, int key);
}
/** @} */
