/* file: dbscan_cl_kernels.cl */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/*
//++
//  Implementation of DBSCAN OpenCL kernels.
//--
*/

#ifndef __DBSCAN_CL_KERNELS_CL__
#define __DBSCAN_CL_KERNELS_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    dbscanClKernels,

    __kernel void computeCores(int numPoints, int numFeatures, algorithmFPType numNbrs, algorithmFPType eps, int useWeights,
                               const __global algorithmFPType * points, const __global algorithmFPType * weights, __global int * cores) {
        const int globalId = get_global_id(0);
        if (get_sub_group_id() > 0) return;

        const int subgroupSize = get_sub_group_size();
        const int localId      = get_sub_group_local_id();
        algorithmFPType count  = 0;
        for (int j = 0; j < numPoints; j++)
        {
            algorithmFPType sum = 0.0;
            for (int i = localId; i < numFeatures; i += subgroupSize)
            {
                algorithmFPType val = points[globalId * numFeatures + i] - points[j * numFeatures + i];
                sum += val * val;
            }
            algorithmFPType distance = sub_group_reduce_add(sum);
            algorithmFPType incr     = (distance <= eps) ? 1.0 : 0.0;
            incr *= useWeights ? weights[globalId] : 1.0;
            count += incr;
        }
        if (localId == 0)
        {
            cores[globalId] = (int)(count >= numNbrs);
        }
    }

    __kernel void startNextCluster(int clusterId, int numPoints, int queueEnd, const __global int * cores, __global int * clusters,
                                   __global int * lastClusterStart, __global int * queue) {
        // The kernel should be run on a single subgroup
        if (get_sub_group_id() > 0 || get_global_id(0) > 0) return;

        const int subgroupSize = get_sub_group_size();
        const int localId      = get_sub_group_local_id();
        const int start        = lastClusterStart[0];
        for (int i = start + localId; i < numPoints; i++)
        {
            const bool found = cores[i] == 1 && clusters[i] < 0;
            const int index  = sub_group_reduce_min(found ? i : numPoints);
            if (index < numPoints)
            {
                if (localId == 0)
                {
                    clusters[index]     = clusterId;
                    lastClusterStart[0] = index + 1;
                    queue[queueEnd]     = index;
                }
                break;
            }
        }
    }

    __kernel void updateQueue(int clusterId, int numPoints, int numFeatures, algorithmFPType eps, int queueStart, int queueEnd,
                              const __global algorithmFPType * points, __global int * cores, __global int * clusters, __global int * queue,
                              __global int * queueFront) {
        if (get_sub_group_id() > 0) return;
        const int subgroupIndex = get_global_id(0);
        if (clusters[subgroupIndex] > -1) return;
        const int localId                  = get_sub_group_local_id();
        const int subgroupSize             = get_sub_group_size();
        volatile __global int * counterPtr = queueFront;

        for (int j = queueStart; j < queueEnd; j++)
        {
            const int index     = queue[j];
            algorithmFPType sum = 0.0;
            for (int i = localId; i < numFeatures; i += subgroupSize)
            {
                algorithmFPType val = points[subgroupIndex * numFeatures + i] - points[index * numFeatures + i];
                sum += val * val;
            }
            algorithmFPType distance = sub_group_reduce_add(sum);
            if (distance > eps) continue;
            if (localId == 0)
            {
                clusters[subgroupIndex] = clusterId;
            }
            if (cores[subgroupIndex] == 0) continue;
            if (localId == 0)
            {
                const int newIndex = atomic_inc(counterPtr);
                queue[newIndex]    = subgroupIndex;
            }
            break;
        }
    }

);

#endif
