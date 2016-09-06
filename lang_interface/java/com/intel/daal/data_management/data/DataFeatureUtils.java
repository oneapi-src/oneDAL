/* file: DataFeatureUtils.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.data_management.data;

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__DATAFEATUREUTILS"></a>
 * @brief Class that provides different feature types
 */
public class DataFeatureUtils {
    private static final int cDaalFloat32 = 0;
    private static final int cDaalFloat64 = 1;
    private static final int cDaalInt32S  = 2;
    private static final int cDaalInt32U  = 3;
    private static final int cDaalInt64S  = 4;
    private static final int cDaalInt64U  = 5;
    private static final int cDaalOtherT  = 10;

    /**
     * Internal data type representing feature value in methods of
     *        the NumericTable class
     */
    static public enum IndexNumType {
        DAAL_FLOAT32(cDaalFloat32), DAAL_FLOAT64(cDaalFloat64), DAAL_INT32_S(cDaalInt32S), DAAL_INT32_U(
                cDaalInt32U), DAAL_INT64_S(cDaalInt64S), DAAL_INT64_U(cDaalInt64U), DAAL_OTHER_T(cDaalOtherT);

        private final int numType;

        IndexNumType(int numType) {
            this.numType = numType;
        }

        public int getType() {
            return numType;
        }
    }

    static final int cDaalSingle = 0;
    static final int cDaalDouble = 1;
    static final int cDaalInt32  = 2;
    static final int cDaalInt64  = 3;
    static final int cDaalOther  = 0xfffffff;

    static int getTypeIndex(Class<?> cls) {
        if (cls == Float.class || cls == float.class) {
            return cDaalSingle;
        } else
        if (cls == Double.class || cls == double.class) {
            return cDaalDouble;
        } else
        if (cls == Integer.class || cls == int.class) {
            return cDaalInt32;
        } else
        if (cls == Long.class || cls == long.class) {
            return cDaalInt64;
        } else {
            return cDaalOther;
        }
    }

    /**
     * Internal data type representing feature value in methods of
     *        the NumericTable class
     */
    static public enum InternalNumType {
        DAAL_SINGLE(cDaalSingle), DAAL_DOUBLE(cDaalDouble), DAAL_INT32(cDaalInt32), DAAL_OTHER(cDaalOther);

        private final int numType;

        InternalNumType(int numType) {
            this.numType = numType;
        }

        public int getType() {
            return numType;
        }
    }

    static Class<?> getClassByType(InternalNumType numType) {
        if (numType == InternalNumType.DAAL_SINGLE) {
            return float.class;
        } else
        if (numType == InternalNumType.DAAL_DOUBLE) {
            return double.class;
        } else
        if (numType == InternalNumType.DAAL_INT32) {
            return int.class;
        } else {
            return null;
        }
    }

    private static final int cDaalGenFloat   = 0;
    private static final int cDaalGenDouble  = 1;
    private static final int cDaalGenInteger = 2;
    private static final int cDaalGenBoolean = 3;
    private static final int cDaalGenString  = 4;
    private static final int cDaalGenUnknown = 0xfffffff;

    /**
     * The Predictive Model Markup Language(PMML) data type
     */
    static public enum PMMLNumType {
        DAAL_GEN_FLOAT(cDaalGenFloat), DAAL_GEN_DOUBLE(cDaalGenDouble), DAAL_GEN_INTEGER(
                cDaalGenInteger), DAAL_GEN_BOOLEAN(cDaalGenBoolean), DAAL_GEN_STRING(cDaalGenString), DAAL_GEN_UNKNOWN(
                        cDaalGenUnknown);

        private final int numType;

        PMMLNumType(int numType) {
            this.numType = numType;
        }

        public int getType() {
            return numType;
        }
    }

    private static final int cDaalCategorical = 0;
    private static final int cDaalOrdinal     = 1;
    private static final int cDaalContinuous  = 2;

    /**
     * Feature type
     */
    static public enum FeatureType {
        DAAL_CATEGORICAL(cDaalCategorical), DAAL_ORDINAL(cDaalOrdinal), DAAL_CONTINUOUS(cDaalContinuous);

        private final int numType;

        FeatureType(int numType) {
            this.numType = numType;
        }

        public int getType() {
            return numType;
        }
    }

    /**
     * @brief @private Casts data from the array of primitive types to the NIO buffer
     */
    interface VectorUpCastIface {
        /**
         * Casts contiguous block of data from the array of primitive types to the NIO
         * buffer
         *
         * @param n     Number of elements to copy from the array to the NIO buffer
         * @param shift Index of the starting element to copy from the array to
         *            the NIO buffer
         * @param src   Input array of primitive types
         * @param dst   Resulting NIO buffer
         */
        void upCast(int n, int shift, Object src, Buffer dst);

        /**
         * Casts non-contiguous block of data from the array of primitive types to the
         * NIO buffer
         *
         * @param n     Number of elements to copy from the array to the NIO buffer
         * @param shift Index of the starting element to copy from the array to NIO buffer
         * @param step  Step between the elements in input array
         * @param src   Input array of primitive types
         * @param dst   Resulting NIO buffer
         */
        void upCastWithStride(int n, int shift, int step, Object src, Buffer dst);

        /**
         * Casts non-contiguous block of data from the array of primitive types to the
         * NIO buffer
         *
         * @param n             Number of elements to copy from the array into NIO buffer
         * @param dataShift     Index of the starting element in input array
         * @param bufferShift   Index of the starting element in output NIO buffer
         * @param bufferStep    Step between the elements in NIO buffer
         * @param src           Input array of primitive types
         * @param dst           Resulting NIO buffer
         */
        void upCastWithBufferStride(int n, int dataShift, int bufferShift, int bufferStep, Object src, Buffer dst);

        /**
         * Casts non-contiguous block of data from the array of primitive types to the
         * NIO buffer with stride
         *
         * @param n             Number of elements to copy from the array into NIO buffer
         * @param dataShift     Index of the starting element in input array
         * @param bufferShift   Index of the starting element in output NIO buffer
         * @param dataStep      Step between the elements in input array
         * @param bufferStep    Step between the elements in NIO buffer
         * @param src           Input array of primitive types
         * @param dst           Resulting NIO buffer
         */
        void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep, Object src,
                Buffer dst);
    }

    /**
     * @brief @private Casts data from the NIO buffer to the array of primitive types
     */
    interface VectorDownCastIface {
        /**
         * Casts continuous block of data from the NIO buffer to the array of
         * primitive types
         *
         * @param n     Number of elements to copy from the NIO buffer into array
         * @param shift Index of the starting element to copy
         * @param src   Input NIO buffer
         * @param dst   Resulting array of primitive types
         */
        void downCast(int n, int shift, Buffer src, Object dst);

        /**
         * Casts non-contiguous block of data from the NIO buffer to the array of
         * primitive types
         *
         * @param n     Number of elements to copy from the NIO buffer into array
         * @param shift Index of the starting element to copy
         * @param step  Step between the elements
         * @param src   Input NIO buffer
         * @param dst   Resulting array of primitive types
         */
        void downCastWithStride(int n, int shift, int step, Buffer src, Object dst);

        /**
         * Casts non-contiguous block of data from the NIO buffer to the continuous
         * array of primitive types
         *
         * @param n             Number of elements to copy from NIO buffer into array
         * @param dataShift     Index of the starting element to put data into array
         * @param bufferShift   Index of the starting element to copy from NIO buffer
         * @param bufferStep    Step between the elements in NIO buffer
         * @param src           Input NIO buffer
         * @param dst           Resulting array of primitive types
         */
        void downCastWithBufferStride(int n, int dataShift, int bufferShift, int bufferStep, Buffer src, Object dst);

        /**
         * Casts non-contiguous block of data from the NIO buffer to the continuous
         * array of primitive types
         *
         * @param n             Number of elements to copy from NIO buffer into array
         * @param dataShift     Index of the starting element to put data into array
         * @param bufferShift   Index of the starting element to copy from NIO buffer
         * @param dataStep      Step between the elements in resulting array
         * @param bufferStep    Step between the elements in NIO buffer
         * @param src           Input NIO buffer
         * @param dst           Resulting array of primitive types
         */
        void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep, Buffer src,
                Object dst);
    }

    /** @private */
    static abstract class VectorUpCastBase implements VectorUpCastIface, Serializable {
        @Override
        public void upCastWithStride(int n, int dataShift, int dataStep, Object src, Buffer dst) {
            upCastWithBothStrides(n, dataShift, 0, dataStep, 1, src, dst);
        }

        @Override
        public void upCastWithBufferStride(int n, int dataShift, int bufferShift, int bufferStep, Object src,
                Buffer dst) {
            upCastWithBothStrides(n, dataShift, bufferShift, 1, bufferStep, src, dst);
        }
    }

    /** @private */
    static abstract class VectorDownCastBase implements VectorDownCastIface, Serializable {
        @Override
        public void downCastWithStride(int n, int dataShift, int dataStep, Buffer src, Object dst) {
            downCastWithBothStrides(n, dataShift, 0, dataStep, 1, src, dst);
        }

        @Override
        public void downCastWithBufferStride(int n, int dataShift, int bufferShift, int bufferStep, Buffer src,
                Object dst) {
            downCastWithBothStrides(n, dataShift, bufferShift, 1, bufferStep, src, dst);
        }
    }

    /** @private */
    static abstract class VectorUpCastBase2Types extends VectorUpCastBase {
        @Override
        public void upCast(int n, int shift, Object src, Buffer dst) {
            upCastWithBothStrides(n, shift, 0, 1, 1, src, dst);
        }
    }

    /** @private */
    static abstract class VectorDownCastBase2Types extends VectorDownCastBase {
        @Override
        public void downCast(int n, int shift, Buffer src, Object dst) {
            downCastWithBothStrides(n, shift, 0, 1, 1, src, dst);
        }
    }

    /** @private */
    static class VectorUpCast {
        static public VectorUpCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return vectorUpCasts[getTypeIndex(fromCls)][getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class VectorDownCast {
        static public VectorDownCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return vectorDownCasts[getTypeIndex(fromCls)][getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class VectorUpCastDouble2Double extends VectorUpCastBase {
        @Override
        public void upCast(int n, int shift, Object src, Buffer dst) {
            dst.position(0);
            ((DoubleBuffer) dst).put((double[]) src, shift, n);
            dst.position(0);
        }

        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            double[] data = (double[]) src;
            for (int i = 0; i < n; i++) {
                ((DoubleBuffer) dst).put(bufferShift + i * bufferStep, data[dataShift + i * dataStep]);
            }
        }
    }

    /** @private */
    static class VectorDownCastDouble2Double extends VectorDownCastBase {
        @Override
        public void downCast(int n, int shift, Buffer src, Object dst) {
            src.position(0);
            double[] data = (double[]) dst;
            ((DoubleBuffer) src).get(data, shift, n);
            src.position(0);
        }

        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            double[] data = (double[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = ((DoubleBuffer) src).get(bufferShift + i * bufferStep);
            }
        }
    }

    /** @private */
    static class VectorUpCastFloat2Double extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            float[] data = (float[]) src;
            for (int i = 0; i < n; i++) {
                ((DoubleBuffer) dst).put(bufferShift + i * bufferStep, (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastFloat2Double extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            double[] data = (double[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (((FloatBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastLong2Double extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            long[] data = (long[]) src;
            for (int i = 0; i < n; i++) {
                ((DoubleBuffer) dst).put(bufferShift + i * bufferStep, (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorUpCastInt2Double extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            int[] data = (int[]) src;
            for (int i = 0; i < n; i++) {
                ((DoubleBuffer) dst).put(bufferShift + i * bufferStep, (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastInt2Double extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            double[] data = (double[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (((IntBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastDouble2Float extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            double[] data = (double[]) src;
            for (int i = 0; i < n; i++) {
                ((FloatBuffer) dst).put(bufferShift + i * bufferStep, (float) (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastDouble2Float extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            float[] data = (float[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (float) (((DoubleBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastFloat2Float extends VectorUpCastBase {
        @Override
        public void upCast(int n, int shift, Object src, Buffer dst) {
            dst.position(0);
            ((FloatBuffer) dst).put((float[]) src, shift, n);
            dst.position(0);
        }

        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            float[] data = (float[]) src;
            for (int i = 0; i < n; i++) {
                ((FloatBuffer) dst).put(bufferShift + i * bufferStep, data[dataShift + i * dataStep]);
            }
        }
    }

    /** @private */
    static class VectorDownCastFloat2Float extends VectorDownCastBase {
        @Override
        public void downCast(int n, int shift, Buffer src, Object dst) {
            src.position(0);
            float[] data = (float[]) dst;
            ((FloatBuffer) src).get(data, shift, n);
            src.position(0);
        }

        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            float[] data = (float[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = ((FloatBuffer) src).get(bufferShift + i * bufferStep);
            }
        }
    }

    /** @private */
    static class VectorUpCastLong2Float extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            long[] data = (long[]) src;
            for (int i = 0; i < n; i++) {
                ((FloatBuffer) dst).put(bufferShift + i * bufferStep, (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorUpCastInt2Float extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            int[] data = (int[]) src;
            for (int i = 0; i < n; i++) {
                ((FloatBuffer) dst).put(bufferShift + i * bufferStep, (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastInt2Float extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            float[] data = (float[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (((IntBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastDouble2Int extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            double[] data = (double[]) src;
            for (int i = 0; i < n; i++) {
                ((IntBuffer) dst).put(bufferShift + i * bufferStep, (int) (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastDouble2Int extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            int[] data = (int[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (int) (((DoubleBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastFloat2Int extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            float[] data = (float[]) src;
            for (int i = 0; i < n; i++) {
                ((IntBuffer) dst).put(bufferShift + i * bufferStep, (int) (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorDownCastFloat2Int extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            int[] data = (int[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (int) (((FloatBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorUpCastLong2Int extends VectorUpCastBase2Types {
        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            long[] data = (long[]) src;
            for (int i = 0; i < n; i++) {
                ((IntBuffer) dst).put(bufferShift + i * bufferStep, (int) (data[dataShift + i * dataStep]));
            }
        }
    }

    /** @private */
    static class VectorUpCastInt2Int extends VectorUpCastBase {
        @Override
        public void upCast(int n, int shift, Object src, Buffer dst) {
            dst.position(0);
            ((IntBuffer) dst).put((int[]) src, shift, n);
            dst.position(0);
        }

        @Override
        public void upCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Object src, Buffer dst) {
            int[] data = (int[]) src;
            for (int i = 0; i < n; i++) {
                ((IntBuffer) dst).put(bufferShift + i * bufferStep, data[dataShift + i * dataStep]);
            }
        }
    }

    /** @private */
    static class VectorDownCastInt2Int extends VectorDownCastBase {
        @Override
        public void downCast(int n, int shift, Buffer src, Object dst) {
            src.position(0);
            int[] data = (int[]) dst;
            ((IntBuffer) src).get(data, shift, n);
            src.position(0);
        }

        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            int[] data = (int[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = ((IntBuffer) src).get(bufferShift + i * bufferStep);
            }
        }
    }

    /** @private */
    static class VectorDownCastDouble2Long extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            long[] data = (long[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (long) (((DoubleBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorDownCastFloat2Long extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            long[] data = (long[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (long) (((FloatBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    /** @private */
    static class VectorDownCastInt2Long extends VectorDownCastBase2Types {
        @Override
        public void downCastWithBothStrides(int n, int dataShift, int bufferShift, int dataStep, int bufferStep,
                Buffer src, Object dst) {
            long[] data = (long[]) dst;
            for (int i = 0; i < n; i++) {
                data[dataShift + i * dataStep] = (((IntBuffer) src).get(bufferShift + i * bufferStep));
            }
        }
    }

    static VectorUpCastIface vectorUpCasts[][] = {
        {(VectorUpCastIface)(new VectorUpCastFloat2Float()),  (VectorUpCastIface)(new VectorUpCastFloat2Double()),
         (VectorUpCastIface)(new VectorUpCastFloat2Int())},
        {(VectorUpCastIface)(new VectorUpCastDouble2Float()), (VectorUpCastIface)(new VectorUpCastDouble2Double()),
         (VectorUpCastIface)(new VectorUpCastDouble2Int())},
        {(VectorUpCastIface)(new VectorUpCastInt2Float()),    (VectorUpCastIface)(new VectorUpCastInt2Double()),
         (VectorUpCastIface)(new VectorUpCastInt2Int())},
        {(VectorUpCastIface)(new VectorUpCastLong2Float()),   (VectorUpCastIface)(new VectorUpCastLong2Double()),
         (VectorUpCastIface)(new VectorUpCastLong2Int())}
    };

    static VectorDownCastIface vectorDownCasts[][] = {
        {(VectorDownCastIface)(new VectorDownCastFloat2Float()),  (VectorDownCastIface)(new VectorDownCastFloat2Double()),
         (VectorDownCastIface)(new VectorDownCastFloat2Int()),  (VectorDownCastIface)(new VectorDownCastFloat2Long())},
        {(VectorDownCastIface)(new VectorDownCastDouble2Float()), (VectorDownCastIface)(new VectorDownCastDouble2Double()),
         (VectorDownCastIface)(new VectorDownCastDouble2Int()), (VectorDownCastIface)(new VectorDownCastDouble2Long())},
        {(VectorDownCastIface)(new VectorDownCastInt2Float()),    (VectorDownCastIface)(new VectorDownCastInt2Double()),
         (VectorDownCastIface)(new VectorDownCastInt2Int()),    (VectorDownCastIface)(new VectorDownCastInt2Long())},
    };
}
