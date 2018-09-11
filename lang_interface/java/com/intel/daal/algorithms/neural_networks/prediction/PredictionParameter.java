/* file: PredictionParameter.java */
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

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PREDICTIONPARAMETER"></a>
 * \brief Class representing the parameters of neural network
 */
public class PredictionParameter extends com.intel.daal.algorithms.Parameter {
    /**
     * Constructs the parameters of neural network algorithm
     * @param context   Context to manage the parameter object
     */
    public PredictionParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public PredictionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the size of the batch to be processed by the neural network
     */
    public long getBatchSize() {
        return cGetBatchSize(cObject);
    }

    /**
     *  Sets the size of the batch to be processed by the neural network
     *  @param batchSize Size of the batch to be processed by the neural network
     */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(cObject, batchSize);
    }

    private native long cInit();
    private native long cGetBatchSize(long cParameter);
    private native void cSetBatchSize(long cParameter, long batchSize);
}
/** @} */
