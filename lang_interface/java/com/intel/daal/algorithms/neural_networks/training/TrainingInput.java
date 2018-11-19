/* file: TrainingInput.java */
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
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGINPUT"></a>
 * @brief %Input object for the training layer
 */
public class TrainingInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets input object for the neural network training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TrainingInputId id, Tensor val) {
        if (id == TrainingInputId.data || id == TrainingInputId.groundTruth) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }
    }

    /**
     * Sets input object for the neural network training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TrainingInputCollectionId id, KeyValueDataCollection val) {
        if (id == TrainingInputCollectionId.groundTruthCollection) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }
    }

    /**
     * Sets input object for the neural network training algorithm
     * @param id    Identifier of the input object
     * @param key   Key to use to retrieve data
     * @param val   Value of the input object
     */
    public void add(TrainingInputCollectionId id, int key, Tensor val) {
        if (id == TrainingInputCollectionId.groundTruthCollection) {
            cAddTensor(this.cObject, id.getValue(), key, val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }
    }

    /**
     * Returns input object for the neural network training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(TrainingInputId id) {
        if (id == TrainingInputId.data || id == TrainingInputId.groundTruth) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns input object for the neural network training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(TrainingInputCollectionId id) {
        if (id == TrainingInputCollectionId.groundTruthCollection) {
            return new KeyValueDataCollection(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns input object for the neural network training algorithm
     * @param  id   Identifier of the result
     * @param  key  Key to use to retrieve data
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(TrainingInputCollectionId id, int key) {
        if (id == TrainingInputCollectionId.groundTruthCollection) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetTensor(this.cObject, id.getValue(), key));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);

    private native void cAddTensor(long cInput, int id, int key, long ntAddr);
    private native long cGetTensor(long cInput, int id, int key);
}
/** @} */
