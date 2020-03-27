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

    void point_to_nbr_distance(__global const algorithmFPType * points, int point_id, int nbr_id, int power, 
                                    int local_id, int subgroup_size, int num_features, __global algorithmFPType * dist)
    {
        algorithmFPType sum     = 0.0;
        for (int i = local_id; i < num_features; i += subgroup_size)
        {
            algorithmFPType val = fabs(points[point_id * num_features + i] - points[nbr_id * num_features + i]);
            sum += pown(val, power);
        }
        algorithmFPType cur_nbr_distance = sub_group_reduce_add(sum);
        if (local_id == 0)
        {
            dist[nbr_id] = cur_nbr_distance;
        }
    }    

    __kernel void compute_point_distances(__global const algorithmFPType * points, int point_id, int power, int num_features, int num_points,
                                          __global algorithmFPType * dist) {
        const int subgroup_index = get_global_id(0) * get_max_sub_group_size() + get_sub_group_id();
        if (subgroup_index >= num_points) return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        point_to_nbr_distance(points, point_id, subgroup_index, power, local_id, subgroup_size, num_features, dist);
    }

    __kernel void compute_queue_block_distances(__global const algorithmFPType * points, __global const int * queue, int queue_begin, int queue_size,
                                                int power, int num_features, int num_points, __global algorithmFPType * queue_dist) {
        const int queue_pos = get_global_id(0);
        if (queue_pos >= queue_size) return;

        const int point_id                        = queue[queue_begin + queue_pos];
        const int group_num                       = get_num_sub_groups();
        const int group_id                        = get_sub_group_id();
        const int subgroup_size                   = get_sub_group_size();
        const int local_id                        = get_sub_group_local_id();
        __global algorithmFPType * cur_block_dist = &queue_dist[queue_pos * num_points];
        for (int j = group_id; j < num_points; j += group_num)
        {
            point_to_nbr_distance(points, point_id, j, power, local_id, subgroup_size, num_features, cur_block_dist);
        }
    }

    __kernel void count_neighbors_by_type(__global const int * assignments, __global const algorithmFPType * distances, int point_id,
                                          int first_chunk_offset, int chunk_size, int num_points, algorithmFPType eps, __global const int * queue,
                                          __global int * counters_all_nbrs, __global int * counters_undef_nbrs) {
        const int subgroup_index        = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int subgroup_offset       = subgroup_index * chunk_size;
        const int subgroup_point_number = num_points - subgroup_offset < chunk_size ? num_points - subgroup_offset : chunk_size;
        point_id                        = first_chunk_offset < 0 ? point_id : queue[point_id];
        const int distance_offset       = subgroup_offset + (first_chunk_offset < 0 ? 0 : first_chunk_offset);
        if (subgroup_index >= num_points) return;

        const int subgroup_size                             = get_sub_group_size();
        const int local_id                                  = get_sub_group_local_id();
        __global const algorithmFPType * subgroup_distances = &distances[distance_offset];
        __global const int * subgroup_assignments           = &assignments[subgroup_offset];
        int counter_all                                     = 0;
        int counter_undefined                               = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int is_nbr           = subgroup_distances[i] <= eps ? 1 : 0;
            int is_undefined_nbr = is_nbr > 0 && subgroup_assignments[i] == _UNDEFINED_ && (i + subgroup_offset != point_id) ? 1 : 0;
            counter_all += is_nbr;
            counter_undefined += is_undefined_nbr;
        }
        int subgroup_counter_all       = sub_group_reduce_add(counter_all);
        int subgroup_counter_undefined = sub_group_reduce_add(counter_undefined);
        if (local_id == 0)
        {
            counters_all_nbrs[subgroup_index]   = subgroup_counter_all;
            counters_undef_nbrs[subgroup_index] = subgroup_counter_undefined;
        }
    }

    __kernel void set_buffer_value(int value_index, int value, __global int * buffer) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[value_index] = value;
    }

    __kernel void set_buffer_value_by_queue_index(__global const int * queue, int queue_index, int value, __global int * buffer) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[queue[queue_index]] = value;
    }

    __kernel void compute_cnunk_offsets(__global const int * chunk_counters, int num_offsets, __global int * chunk_offsets) {
        if (get_global_id(0) > 0 || get_sub_group_id() > 0) return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        int subgroup_offset     = 0;
        for (int i = local_id; i < num_offsets; i += subgroup_size)
        {
            int cur_counter  = chunk_counters[i];
            int local_offset = subgroup_offset + sub_group_scan_exclusive_add(cur_counter);
            int total_offset = sub_group_reduce_add(cur_counter);
            chunk_offsets[i] = local_offset;
            subgroup_offset += total_offset;
        }
    }

    __kernel void push_points_to_queue(__global const algorithmFPType * distances, __global const int * chunk_offsets, int queue_end, int point_id,
                                       int cluster_id, int first_chunk_offset, int chunk_size, algorithmFPType eps, int num_points,
                                       __global int * assignments, __global int * queue) {
        const int subgroup_index  = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int subgroup_offset = subgroup_index * chunk_size;
        point_id                  = first_chunk_offset < 0 ? point_id : queue[point_id];
        const int dist_offset     = subgroup_offset + (first_chunk_offset < 0 ? 0 : first_chunk_offset);
        if (subgroup_offset >= num_points) return;

        const int subgroup_point_number                     = num_points - subgroup_offset < chunk_size ? num_points - subgroup_offset : chunk_size;
        const int subgroup_size                             = get_sub_group_size();
        const int local_id                                  = get_sub_group_local_id();
        __global const algorithmFPType * subgroup_distances = &distances[dist_offset];
        __global int * subgroup_assignments                 = &assignments[subgroup_offset];
        const int out_offset                                = chunk_offsets[subgroup_index];
        int local_offset                                    = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int is_nbr           = subgroup_distances[i] <= eps ? 1 : 0;
            int is_undefined_nbr = is_nbr > 0 && subgroup_assignments[i] == _UNDEFINED_ && (i + subgroup_offset != point_id) ? 1 : 0;
            int local_pos        = sub_group_scan_exclusive_add(is_undefined_nbr);
            if (is_undefined_nbr)
            {
                subgroup_assignments[i]                                  = cluster_id;
                queue[queue_end + out_offset + local_offset + local_pos] = subgroup_offset + i;
            }
            if (is_nbr && (subgroup_assignments[i] == _NOISE_))
            {
                subgroup_assignments[i] = cluster_id;
            }
            local_offset += sub_group_reduce_add(is_undefined_nbr);
        }
        if (subgroup_index == 0 && local_id == 0)
        {
            assignments[point_id] = cluster_id;
        }
    }

);

#endif
