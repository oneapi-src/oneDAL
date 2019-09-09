/* file: TrainingModel.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.ForwardLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.BackwardLayers;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.neural_networks.NextLayersCollection;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.prediction.PredictionModel;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGMODEL"></a>
 * @brief Class Model object for the training stage of neural network
 */
public class TrainingModel extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs model for the training stage of the neural network
     * @param context    Context to manage the model
     */
    public TrainingModel(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    /**
     * Constructs model for the training stage of the neural network by copying parameters of another model
     * @param context    Context to manage the model
     * @param other      A model to be used as the source to initialize the parameters of the model
     */
    public TrainingModel(DaalContext context, TrainingModel other) {
        super(context);
        cObject = cInit(other.cObject);
    }

    public TrainingModel(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Initializes neural network
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     * @param sampleSize  Dimensionality of the training data
     * @param topology  Collection of layer descriptors of every inserted layer
     */
    public void initialize(Class <? extends Number> cls, long[] sampleSize, TrainingTopology topology) {
        if (cls == Double.class) {
            cInitialize(cObject, Precision.doublePrecision.getValue(), sampleSize, topology.cObject);
        } else {
            cInitialize(cObject, Precision.singlePrecision.getValue(), sampleSize, topology.cObject);
        }
    }

    /**
     * Returns list of forward layers
     * @return List of forward layers
     */
    public ForwardLayers getForwardLayers() {
        return new ForwardLayers(getContext(), cGetForwardLayers(cObject));
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * @param index  Index of the layer in the network
     * @return Forward stage of a layer with certain index in the network
     */
    public ForwardLayer getForwardLayer(long index) {
        return new ForwardLayer(getContext(), cGetForwardLayer(cObject, index));
    }

    /**
     * Returns list of backward layers
     * @return List of backward layers
     */
    public BackwardLayers getBackwardLayers() {
        return new BackwardLayers(getContext(), cGetBackwardLayers(cObject));
    }

    /**
     * Returns the backward stage of a layer with certain index in the network
     * @param index  Index of the layer in the network
     * @return Backward stage of a layer with certain index in the network
     */
    public BackwardLayer getBackwardLayer(long index) {
        return new BackwardLayer(getContext(), cGetBackwardLayer(cObject, index));
    }

    /**
     * Returns list of connections between layers
     * @return List of next layers for each layer with corresponding index
     */
    public NextLayersCollection getNextLayers() {
        return new NextLayersCollection(getContext(), cGetNextLayers(cObject));
    }

    /**
     * Returns list of forward layers and their parameters organised in the PredictionModel
     * @param cls               Data type to use in intermediate computations for the neural network,
     *                          Double.class or Float.class
     * \return List of forward layers and their parameters organised in the PredictionModel
     */
    public PredictionModel getPredictionModel(Class <? extends Number> cls) {
        if (cls == Double.class) {
            return new PredictionModel(getContext(), cGetPredictionModel(Precision.doublePrecision.getValue(), cObject));
        } else {
            return new PredictionModel(getContext(), cGetPredictionModel(Precision.singlePrecision.getValue(), cObject));
        }
    }

    /**
     * Returns the weights and biases of the forward layer of neural network as numeric table
     * @return The weights and biases numeric table
     */
    public NumericTable getWeightsAndBiases() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetWeightsAndBiases(cObject));
    }

    /**
     * Sets the weights and biases of the forward layer of neural network as numeric table
     * @param weightsAndBiases  The weights and biases numeric table
     */
    public void setWeightsAndBiases(NumericTable weightsAndBiases) {
        cSetWeightsAndBiases(cObject, weightsAndBiases.getCObject());
    }

    private native long cInit();
    private native long cInit(long cModel);
    private native void cInitialize(long cModel, int prec, long[] sampleSize, long layerDescriptorAddr);
    private native long cGetForwardLayers(long cModel);
    private native long cGetForwardLayer(long cModel, long index);
    private native long cGetBackwardLayers(long cModel);
    private native long cGetBackwardLayer(long cModel, long index);
    private native long cGetNextLayers(long cModel);
    private native long cGetPredictionModel(int prec, long cModel);
    private native long cGetWeightsAndBiases(long cModel);
    private native void cSetWeightsAndBiases(long cModel, long nt);
}
/** @} */
