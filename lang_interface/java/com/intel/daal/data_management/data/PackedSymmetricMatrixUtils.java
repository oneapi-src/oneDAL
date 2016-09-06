/* file: PackedSymmetricMatrixUtils.java */
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;

class PackedSymmetricMatrixUtils {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    interface SymmetricAccessIface {
        int getPosition(int row, int column, int nDim);
    }

    static class SymmetricAccess {
        static public SymmetricAccessIface getAccess(NumericTable.StorageLayout packedLayout) {
            if (packedLayout == NumericTable.StorageLayout.upperPackedSymmetricMatrix) {
                return symmetricAccessList[0];
            } else {
                return symmetricAccessList[1];
            }
        }
    }

    static class UpperSymmetricAccess implements SymmetricAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row > column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 * nDim - 1 * (row - 1)) * row) / 2;
            int colStartOffset = column - row;
            return rowStartOffset + colStartOffset;
       }
    }

    static class LowerSymmetricAccess implements SymmetricAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row < column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 + 1 * (row - 1)) * row) / 2;
            int colStartOffset = column;
            return rowStartOffset + colStartOffset;
        }
    }

    interface SymmetricUpCastIface {
        void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout);
        void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout);
    }

    interface SymmetricDownCastIface {
        void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout);
        void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout);
    }

    /** @private */
    static class SymmetricUpCast {
        static public SymmetricUpCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return symmetricUpCasts[DataFeatureUtils.getTypeIndex(fromCls)][DataFeatureUtils.getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class SymmetricDownCast {
        static public SymmetricDownCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return symmetricDownCasts[DataFeatureUtils.getTypeIndex(fromCls)][DataFeatureUtils.getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            SymmetricAccessIface symmetricAccess = SymmetricAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }

    static SymmetricUpCastIface symmetricUpCasts[][] = {
        {(SymmetricUpCastIface)(new SymmetricUpCastFloat2Float()), (SymmetricUpCastIface)(new SymmetricUpCastFloat2Double()),
         (SymmetricUpCastIface)(new SymmetricUpCastFloat2Int())},
        {(SymmetricUpCastIface)(new SymmetricUpCastDouble2Float()), (SymmetricUpCastIface)(new SymmetricUpCastDouble2Double()),
         (SymmetricUpCastIface)(new SymmetricUpCastDouble2Int())},
        {(SymmetricUpCastIface)(new SymmetricUpCastInt2Float()), (SymmetricUpCastIface)(new SymmetricUpCastInt2Double()),
         (SymmetricUpCastIface)(new SymmetricUpCastInt2Int())},
        {(SymmetricUpCastIface)(new SymmetricUpCastLong2Float()), (SymmetricUpCastIface)(new SymmetricUpCastLong2Double()),
         (SymmetricUpCastIface)(new SymmetricUpCastLong2Int())}
    };

    static SymmetricDownCastIface symmetricDownCasts[][] = {
        {(SymmetricDownCastIface)(new SymmetricDownCastFloat2Float()), (SymmetricDownCastIface)(new SymmetricDownCastFloat2Double()),
         (SymmetricDownCastIface)(new SymmetricDownCastFloat2Int()), (SymmetricDownCastIface)(new SymmetricDownCastFloat2Long())},
        {(SymmetricDownCastIface)(new SymmetricDownCastDouble2Float()), (SymmetricDownCastIface)(new SymmetricDownCastDouble2Double()),
         (SymmetricDownCastIface)(new SymmetricDownCastDouble2Int()), (SymmetricDownCastIface)(new SymmetricDownCastDouble2Long())},
        {(SymmetricDownCastIface)(new SymmetricDownCastInt2Float()), (SymmetricDownCastIface)(new SymmetricDownCastInt2Double()),
         (SymmetricDownCastIface)(new SymmetricDownCastInt2Int()), (SymmetricDownCastIface)(new SymmetricDownCastInt2Long())},
    };

    static SymmetricAccessIface symmetricAccessList[] = {
        (SymmetricAccessIface)(new UpperSymmetricAccess()),
        (SymmetricAccessIface)(new LowerSymmetricAccess())
    };
}
