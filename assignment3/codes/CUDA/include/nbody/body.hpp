//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>
#include <omp.h>

class BodyPool;

__device__
int getBlockId() {
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__
int getLocalThreadId() {
  return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__
int getThreadId() {
  int blockId = getBlockId();
  int localThreadId = getLocalThreadId();
  return blockId * (blockDim.x * blockDim.y * blockDim.z) + localThreadId;
}

__global__
void cuda_update_for_tick(double elapse,
                          double gravity,
                          double position_range,
                          double radius,
                          int totalThreadSize,
                          BodyPool * pool);

__global__
void test(int testNum);


class BodyPool {
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.01;

public:
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
    double * x;
    double * y;
    double * vx;
    double * vy;
    double * ax;
    double * ay;
    double * m;
    int _size;

    class Body {
        int index;
        BodyPool &pool;

        friend class BodyPool;

        __device__ __host__
        Body(int index, BodyPool &pool) : index(index), pool(pool) {}

    public:
        __device__ __host__
        double &get_x() {
            return pool.x[index];
        }
        __device__ __host__
        double &get_y() {
            return pool.y[index];
        }
        __device__ __host__
        double &get_vx() {
            return pool.vx[index];
        }
        __device__ __host__
        double &get_vy() {
            return pool.vy[index];
        }
        __device__ __host__
        double &get_ax() {
            return pool.ax[index];
        }
        __device__ __host__
        double &get_ay() {
            return pool.ay[index];
        }
        __device__ __host__
        double &get_m() {
            return pool.m[index];
        }
        __device__ __host__
        double distance_square(Body &that) {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }
        __device__ __host__
        double distance(Body &that) {
            return std::sqrt(distance_square(that));
        }
        __device__ __host__
        double delta_x(Body &that) {
            return get_x() - that.get_x();
        }
        __device__ __host__
        double delta_y(Body &that) {
            return get_y() - that.get_y();
        }
        __device__ __host__
        bool collide(Body &that, double radius) {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        __device__ __host__
        void handle_wall_collision(double position_range, double radius) {
            bool flag = false;
            if (get_x() <= radius) {
                flag = true;
                get_x() = radius + radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            } else if (get_x() >= position_range - radius) {
                flag = true;
                get_x() = position_range - radius - radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }

            if (get_y() <= radius) {
                flag = true;
                get_y() = radius + radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            } else if (get_y() >= position_range - radius) {
                flag = true;
                get_y() = position_range - radius - radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            if (flag) {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        __device__ __host__
        void update_for_tick(
                double elapse,
                double position_range,
                double radius) {
            get_vx() += get_ax() * elapse;
            get_vy() += get_ay() * elapse;
            handle_wall_collision(position_range, radius);
            get_x() += get_vx() * elapse;
            get_y() += get_vy() * elapse;
            handle_wall_collision(position_range, radius);
        }

    };
    
    __host__
    BodyPool(size_t size, double position_range, double mass_range){
        _size = size;

        x = (double*)malloc(size*sizeof(double));
        y = (double*)malloc(size*sizeof(double));
        vx = (double*)malloc(size*sizeof(double));
        vy = (double*)malloc(size*sizeof(double));
        ax = (double*)malloc(size*sizeof(double));
        ay = (double*)malloc(size*sizeof(double));
        m = (double*)malloc(size*sizeof(double));

        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        for (int i=0;i<size;i++) {
            x[i] = position_dist(engine);
            y[i] = position_dist(engine);
            m[i] = mass_dist(engine);
            vx[i] = 0;
            vy[i] = 0;
            ax[i] = 0;
            ay[i] = 0;
        }
    }

    __host__
    ~BodyPool(){
        free(x);
        free(y);
        free(vx);
        free(vy);
        free(ax);
        free(ay);
        free(m);
    }
    
    __device__ __host__
    Body get_body(int index) {
        return {index, *this};
    }

    __device__ __host__
    void clear_acceleration() {
        memset(ax, 0, sizeof(ax));
        memset(ay, 0, sizeof(ay));
    }

    __device__ __host__
    int size() {
        return _size;
    }

    __device__ __host__
    static void check_and_update(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        if (i.collide(j, radius)) {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx())
                            + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.get_vx() -= scalar * delta_x * j.get_m();
            i.get_vy() -= scalar * delta_y * j.get_m();
            j.get_vx() += scalar * delta_x * i.get_m();
            j.get_vy() += scalar * delta_y * i.get_m();
            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.get_x() += delta_x / distance * ratio * radius / 2.0;
            i.get_y() += delta_y / distance * ratio * radius / 2.0;
            j.get_x() -= delta_x / distance * ratio * radius / 2.0;
            j.get_y() -= delta_y / distance * ratio * radius / 2.0;
        } else {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_ax() -= scalar * delta_x * j.get_m();
            i.get_ay() -= scalar * delta_y * j.get_m();
            j.get_ax() += scalar * delta_x * i.get_m();
            j.get_ay() += scalar * delta_y * i.get_m();
        }
    }

    __host__
    void update_for_tick(double elapse,
                         double gravity,
                         double position_range,
                         double radius,
                         dim3 grid_size,
                         dim3 block_size,
                         BodyPool * pool) {

        memset(ax, 0, sizeof(ax));
        memset(ay, 0, sizeof(ay));

        int totalThreadSize = grid_size.x * grid_size.y * grid_size.z * block_size.x * block_size.y * block_size.z;

        // allocate device memory
        BodyPool * cudaPoolCopy;
        cudaMallocManaged((void**)&cudaPoolCopy, sizeof(*pool));
        cudaMemcpy(cudaPoolCopy, pool, sizeof(*pool), cudaMemcpyHostToDevice);

        // allocate memory for arrays
        double *xcopy, *ycopy, *vxcopy, *vycopy, *axcopy, *aycopy, *mcopy;
        cudaMallocManaged((void**) &(xcopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(ycopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(vxcopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(vycopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(axcopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(aycopy), sizeof(double)*pool->size());
        cudaMallocManaged((void**) &(mcopy), sizeof(double)*pool->size());

        // copy values to arrays
        cudaMemcpy(xcopy, pool->x, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ycopy, pool->y, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(vxcopy, pool->vx, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(vycopy, pool->vy, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(axcopy, pool->ax, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(aycopy, pool->ay, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(mcopy, pool->m, sizeof(double)*pool->size(), cudaMemcpyHostToDevice);
        
        cudaPoolCopy->x = xcopy;
        cudaPoolCopy->y = ycopy;
        cudaPoolCopy->vx = vxcopy;
        cudaPoolCopy->vy = vycopy;
        cudaPoolCopy->ax = axcopy;
        cudaPoolCopy->ay = aycopy;
        cudaPoolCopy->m = mcopy;

        cuda_update_for_tick<<<grid_size, block_size>>>(elapse, gravity, position_range, radius, totalThreadSize, cudaPoolCopy);
        cudaDeviceSynchronize();

        for (int i = 0; i < pool->size(); ++i) {
            pool->x[i] = cudaPoolCopy->x[i];
            pool->y[i] = cudaPoolCopy->y[i];
            pool->vx[i] = cudaPoolCopy->vx[i];
            pool->vy[i] = cudaPoolCopy->vy[i];
            pool->ax[i] = cudaPoolCopy->ax[i];
            pool->ay[i] = cudaPoolCopy->ay[i];
        }

        cudaFree(cudaPoolCopy);
        cudaFree(xcopy);
        cudaFree(ycopy);
        cudaFree(vxcopy);
        cudaFree(vycopy);
        cudaFree(axcopy);
        cudaFree(aycopy);
        cudaFree(mcopy);

        // printf("\n\n");
        
    }
};

__global__
void cuda_update_for_tick(double elapse,
                          double gravity,
                          double position_range,
                          double radius,
                          int totalThreadSize,
                          BodyPool * pool) {

    int taskNum = getThreadId();

    // allocate memory for itself
    BodyPool * copyPool;
    cudaMalloc((void**)&copyPool, sizeof(*pool));
    copyPool->_size = pool->size();

    // allocate memory for arrays
    double *xcopy, *ycopy, *vxcopy, *vycopy, *axcopy, *aycopy, *mcopy;
    cudaMalloc((void**) &(xcopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(ycopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(vxcopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(vycopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(axcopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(aycopy), sizeof(double)*pool->size());
    cudaMalloc((void**) &(mcopy), sizeof(double)*pool->size());

    // copy values to arrays
    memcpy(xcopy, pool->x, sizeof(double)*pool->size());
    memcpy(ycopy, pool->y, sizeof(double)*pool->size());
    memcpy(vxcopy, pool->vx, sizeof(double)*pool->size());
    memcpy(vycopy, pool->vy, sizeof(double)*pool->size());
    memcpy(axcopy, pool->ax, sizeof(double)*pool->size());
    memcpy(aycopy, pool->ay, sizeof(double)*pool->size());
    memcpy(mcopy, pool->m, sizeof(double)*pool->size());

    copyPool->x = xcopy;
    copyPool->y = ycopy;
    copyPool->vx = vxcopy;
    copyPool->vy = vycopy;
    copyPool->ax = axcopy;
    copyPool->ay = aycopy;
    copyPool->m = mcopy;

    // make sure all threads finish copy
    __syncthreads();

    int startPosition = (copyPool->size()/totalThreadSize)*taskNum + min(static_cast<int> (copyPool->size())%totalThreadSize, taskNum);
    int endPosition = (copyPool->size()/totalThreadSize)*(taskNum+1) + min(static_cast<int> (copyPool->size())%totalThreadSize, taskNum+1)-1;

    // printf("%d %d %d\n", taskNum, startPosition, endPosition);

    // consider the situations of bodies between that outsides the range and inside the range
    for (int i = 0; i < startPosition; ++i) {
        for (int j = startPosition; j <= endPosition; ++j) {
            copyPool->check_and_update(copyPool->get_body(i), copyPool->get_body(j), radius, gravity);
        }
    }

    for (int i = endPosition+1; i < copyPool->size(); ++i) {
        for (int j = startPosition; j <= endPosition; ++j) {
            copyPool->check_and_update(copyPool->get_body(i), copyPool->get_body(j), radius, gravity);
        }
    }

    // consider the situations of bodies inside the range
    for (int i = startPosition; i <= endPosition; ++i) {
        for (int j = i + 1; j <= endPosition; ++j) {
            copyPool->check_and_update(copyPool->get_body(i), copyPool->get_body(j), radius, gravity);
        }
    }

    // only update the position of bodies inside the range
    for (int i = startPosition; i <= endPosition; ++i) {
        copyPool->get_body(i).update_for_tick(elapse, position_range, radius);
    }

    // update data back to host
    for (int i = startPosition; i <= endPosition; ++i) {
        pool->x[i] = copyPool->x[i];
        pool->y[i] = copyPool->y[i];
        pool->vx[i] = copyPool->vx[i];
        pool->vy[i] = copyPool->vy[i];
        pool->ax[i] = copyPool->ax[i];
        pool->ay[i] = copyPool->ay[i];
    }

    cudaFree(copyPool);
    cudaFree(xcopy);
    cudaFree(ycopy);
    cudaFree(vxcopy);
    cudaFree(vycopy);
    cudaFree(axcopy);
    cudaFree(aycopy);
    cudaFree(mcopy);

}