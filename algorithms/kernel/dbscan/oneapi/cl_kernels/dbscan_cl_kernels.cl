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
    dbscan_cl_kernels,

    __kernel void compute_point_distances(__global const algorithmFPType * points, int point_id, int power, int num_features,
                                  int num_points, __global algorithmFPType * dist) {
        const int subgroup_index = get_global_id(0) * get_max_sub_group_size() + get_sub_group_id();
        if (subgroup_index >= num_points) return;

        const int subgroup_size      = get_sub_group_size();
        const int local_id        = get_sub_group_local_id();
        algorithmFPType sum = 0.0;
        for (int i = local_id; i < num_features; i += subgroup_size)
        {
            algorithmFPType val = fabs(points[point_id * num_features + i] - points[subgroup_index * num_features + i]);
            sum += pown(val, power);
        }
        algorithmFPType cur_nbr_distance = sub_group_reduce_add(sum);
        if (local_id == 0)
        {
            dist[subgroup_index] = cur_nbr_distance;
        }
    }

    __kernel void compute_queue_block_distances(__global const algorithmFPType * points, __global const int * queue,
                                        int queue_begin, int queue_size, int power, int num_features, int num_points, __global algorithmFPType * dist) {
        const int queue_pos = get_global_id(0);
        if (queue_pos >= queue_size) return;

        const int point_id                        = queue[queue_begin + queue_pos];
        const int group_num                 = get_num_sub_groups();
        const int group_id                  = get_sub_group_id();
        const int subgroup_size                = get_sub_group_size();
        const int local_id                  = get_sub_group_local_id();
        __global algorithmFPType * cur_block_dist = &dist[queue_pos * num_points];
        for (int j = group_id; j < num_points; j += group_num)
        {
            algorithmFPType sum = 0.0;
            for (int i = local_id; i < num_features; i += subgroup_size)
            {
                algorithmFPType val = fabs(points[point_id * num_features + i] - points[j * num_features + i]);
                sum += pown(val, power);
            }
            algorithmFPType cur_nbr_distance = sub_group_reduce_add(sum);
            if (local_id == 0)
            {
                cur_block_dist[j] = cur_nbr_distance;
            }
        }
    }

    __kernel void count_neighbors_by_type(__global const int * assignments, __global const algorithmFPType * points, int point_id, int first_chunk_offset,
                                  int chunk_size, int num_points, algorithmFPType eps, __global const int * queue, __global int * counters_all_nbrs,
                                  __global int * counters_undef_nbrs) {
        const int subgroup_index           = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int subgroup_offset          = subgroup_index * chunk_size;
        const int subgroup_point_number          = num_points - subgroup_offset < chunk_size ? num_points - subgroup_offset : chunk_size;
        point_id                    = first_chunk_offset < 0 ? point_id : queue[point_id];
        const int distance_offset = subgroup_offset + (first_chunk_offset < 0 ? 0 : first_chunk_offset);
        if (subgroup_index >= num_points) return;

        const int subgroup_size                         = get_sub_group_size();
        const int local_id                           = get_sub_group_local_id();
        __global const algorithmFPType * subgroup_points = &points[distance_offset];
        __global const int * assignments_assignments          = &assignments[subgroup_offset];
        int counter_all                              = 0;
        int counter_undefined                          = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int is_nbr     = subgroup_points[i] <= eps ? 1 : 0;
            int is_undefined_nbr = is_nbr > 0 && assignments_assignments[i] == _UNDEFINED_ && (i + subgroup_offset != point_id) ? 1 : 0;
            counter_all += is_nbr;
            counter_undefined += is_undefined_nbr;
        }
        int subgroup_counter_all     = sub_group_reduce_add(counter_all);
        int subgroup_counter_undefined = sub_group_reduce_add(counter_undefined);
        if (local_id == 0)
        {
            counters_all_nbrs[subgroup_index]      = subgroup_counter_all;
            counters_undef_nbrs[subgroup_index] = subgroup_counter_undefined;
        }
    }

    __kernel void set_buffer_value(__global int * buffer, int index, int value) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[index] = value;
    } 
    
    __kernel void set_buffer_value_by_queue_index(__global int * queue, __global int * buffer, int pos, int value) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[queue[pos]] = value;
    }

    __kernel void count_offsets(__global const int * counters, __global int * offsets, int num_offsets) {
        if (get_global_id(0) > 0 || get_sub_group_id() > 0) return;
        const int subgroup_size = get_sub_group_size();
        const int local_id   = get_sub_group_local_id();
        int start      = 0;
        for (int i = local_id; i < num_offsets; i += subgroup_size)
        {
            int cur_counter  = counters[i];
            int local_offset = start + sub_group_scan_exclusive_add(cur_counter);
            int total_offset = sub_group_reduce_add(cur_counter);
            offsets[i]       = local_offset;
            start += total_offset;
        }
    }

    __kernel void push_to_queue(__global const algorithmFPType * distances, __global const int * offsets, __global int * assignments,
                                __global int * queue, int queue_end, int point_id, int cluster_id, int chunk_offset, int chunk_size,
                                algorithmFPType eps, int num_points) {
        const int subgroup_index       = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int offset      = subgroup_index * chunk_size;
        point_id                = chunk_offset < 0 ? point_id : queue[point_id];
        const int dist_offset = offset + (chunk_offset < 0 ? 0 : chunk_offset);
        if (offset >= num_points) return;
        const int subgroup_point_number                       = num_points - offset < chunk_size ? num_points - offset : chunk_size;
        const int subgroup_size                         = get_sub_group_size();
        const int local_id                           = get_sub_group_local_id();
        __global const algorithmFPType * input = &distances[dist_offset];
        __global int * assigned                = &assignments[offset];
        const int out_offset                   = offsets[subgroup_index];
        int local_offset                       = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int nbr_flag     = input[i] <= eps ? 1 : 0;
            int new_nbr_flag = nbr_flag > 0 && assigned[i] == _UNDEFINED_ && (i + offset != point_id) ? 1 : 0;
            int pos          = sub_group_scan_exclusive_add(new_nbr_flag);
            if (new_nbr_flag)
            {
                assigned[i]                                        = cluster_id;
                queue[queue_end + out_offset + local_offset + pos] = offset + i;
            }
            if (nbr_flag && (assigned[i] == _NOISE_))
            {
                assigned[i] = cluster_id;
            }
            local_offset += sub_group_reduce_add(new_nbr_flag);
        }
        if (subgroup_index == 0 && local_id == 0)
        {
            assignments[point_id] = cluster_id;
        }
    }

);

#endif
