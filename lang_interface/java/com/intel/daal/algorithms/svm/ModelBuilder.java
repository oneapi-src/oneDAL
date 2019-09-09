/* file: ModelBuilder.java */
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
 * @defgroup svm Support Vector Machine Classifier
 * @brief Contains classes to work with the support vector machine classifier
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.svm;

import com.intel.daal.utils.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL__BUILDER"></a>
 * @brief %Class for building model of the support vector machine algorithm
 *
 * @par References
 *      - Parameter class
 */
public class ModelBuilder extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private Precision prec;

    /**
     * Constructs the support vector machine model builder
     * @param context          Context to manage SVM model builder
     * @param cls              Data type to use in intermediate computations of SVM model data,
     *                         Double.class or Float.class
     * @param nFeatures        Number of features
     * @param nSupportVectors  Number of support vectors in model
     */
    public ModelBuilder(DaalContext context, Class<? extends Number> cls, long nFeatures, long nSupportVectors) {
        super(context);
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), nFeatures, nSupportVectors);
    }

    /**
     * Get built model of SVM
     * @return Model of SVM
     */
    public Model getModel() {
        return new Model(getContext(), cGetModel(this.cObject, prec.getValue()));
    }

    /**
     * Set bias term to model
     * @param bias The bias value
     */
    public void setBias(double bias) {
        cSetBiasDouble(this.cObject, bias);
    }

    /**
     * Set support vectors to the model
     * @param supportVestors    Set of support vectors
     * @param length            Size of set of support-vectors
     */
    public void setSupportVectors(FloatBuffer supportVestors, long length) {
        final long bufferSize = length;
        float[] supportVectorArr = new float[(int)bufferSize];
        supportVestors.position(0);
        supportVestors.get(supportVectorArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(supportVectorArr);

        cSetSupportVectorsFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set support vectors to the model
     * @param supportVestors    Set of support vectors
     * @param length            Size of set of support-vectors
     */
    public void setSupportVectors(DoubleBuffer supportVestors, long length) {
        final long bufferSize = length;
        double[] supportVectorArr = new double[(int)bufferSize];
        supportVestors.position(0);
        supportVestors.get(supportVectorArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(supportVectorArr);

        cSetSupportVectorsDouble(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set support vectors to the model
     * @param supportVestors    Set of support vectors
     */
    public void setSupportVectors(float [] supportVestors) {
        final long bufferSize = supportVestors.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(supportVestors);

        cSetSupportVectorsFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set support vectors to the model
     * @param supportVestors    Set of support vectors
     */
    public void setSupportVectors(double [] supportVestors) {
        final long bufferSize = supportVestors.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(supportVestors);

        cSetSupportVectorsDouble(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set classification coefficients to the model
     * @param classCoefficients Set of classification coefficients
     * @param length            Size of set of classification coefficients
     */
    public void setClassificationCoefficients(FloatBuffer classCoefficients, long length) {
        final long bufferSize = length;
        float[] classCoefficientsArr = new float[(int)bufferSize];
        classCoefficients.position(0);
        classCoefficients.get(classCoefficientsArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(classCoefficientsArr);

        cSetClassificationCoefficientsFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set classification coefficients to the model
     * @param classCoefficients Set of classification coefficients
     * @param length            Size of set of classification coefficients
     */
    public void setClassificationCoefficients(DoubleBuffer classCoefficients, long length) {
        final long bufferSize = length;
        double[] classCoefficientsArr = new double[(int)bufferSize];
        classCoefficients.position(0);
        classCoefficients.get(classCoefficientsArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(classCoefficientsArr);

        cSetClassificationCoefficientsDouble(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set classification coefficients to the model
     * @param classCoefficients Set of classification coefficients
     */
    public void setClassificationCoefficients(float [] classCoefficients) {
        final long bufferSize = classCoefficients.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(classCoefficients);

        cSetClassificationCoefficientsFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set classification coefficients to the model
     * @param classCoefficients Set of classification coefficients
     */
    public void setClassificationCoefficients(double [] classCoefficients) {
        final long bufferSize = classCoefficients.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(classCoefficients);

        cSetClassificationCoefficientsDouble(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Set support indices to the model
     * @param supportIndices    Set of support indices
     * @param length            Size of set of classification coefficients
     */
    public void setSupportIndices(IntBuffer supportIndices, long length) {
        final long bufferSize = length;
        int[] supportIndicesArr = new int[(int)bufferSize];
        supportIndices.position(0);
        supportIndices.get(supportIndicesArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(supportIndicesArr);

        cSetSupportIndices(this.cObject, prec.getValue(), byteBuf, bufferSize);
    }

    /**
     * Set support indices to the model
     * @param supportIndices    Set of support indices
     */
    public void setSupportIndices(int [] supportIndices) {
        final long bufferSize = supportIndices.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(supportIndices);

        cSetSupportIndices(this.cObject, prec.getValue(), byteBuf, bufferSize);
    }

    private native long cInit(int prec, long nFeatures, long nSupportVectors);
    private native long cGetModel(long algAddr, int prec);
    private native void cSetBiasDouble(long algAddr, double bias);
    private native void cSetBiasFloat(long algAddr, float bias);
    private native void cSetSupportVectorsFloat(long algAddr, ByteBuffer byteBuffer, long nValues);
    private native void cSetSupportVectorsDouble(long algAddr, ByteBuffer byteBuffer, long nValues);
    private native void cSetClassificationCoefficientsFloat(long algAddr, ByteBuffer byteBuffer, long nValues);
    private native void cSetClassificationCoefficientsDouble(long algAddr, ByteBuffer byteBuffer, long nValues);
    private native void cSetSupportIndices(long algAddr, int prec, ByteBuffer byteBuffer, long nValues);
}
/** @} */
