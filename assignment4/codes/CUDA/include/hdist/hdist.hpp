#pragma once

namespace hdist {
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

    enum class Algorithm : int {
        Jacobi = 0,
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const{
            return 
            (this->room_size == that.room_size) &&
            (this->block_size == that.block_size) &&
            (this->source_x == that.source_x) &&
            (this->source_y == that.source_y) &&
            (this->source_temp == that.source_temp) &&
            (this->border_temp == that.border_temp) &&
            (this->tolerance == that.tolerance) &&
            (this->sor_constant == that.sor_constant) &&
            (this->algo == that.algo);
        }
    };

    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        double *data0, *data1;
        size_t current_buffer = 0;
        size_t length;

        __host__
        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                : length(size) {
            
            data0 = (double*)malloc(size*size*sizeof(double));
            data1 = (double*)malloc(size*size*sizeof(double));

            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->set(i, j, border_temp);
                    } else if (i == x && j == y) {
                        this->set(i, j, source_temp);
                    } else {
                        this->set(i, j, 0);
                    }
                }
            }
        }

        __host__ __device__
        double* get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        __host__ __device__
        double fetch(int index1, int index2) {
            return get_current_buffer()[index1 * length + index2];
        }

        __host__ __device__
        double fetch(Alt alt, int index1, int index2) {
            return current_buffer == 1 ? data0[index1 * length + index2] : data1[
                    index1 * length + index2];
        }

        __host__ __device__
        void set(int index1, int index2, double setVal) {
            get_current_buffer()[index1 * length + index2] = setVal;
        }

        __host__ __device__
        void set(Alt alt, int index1, int index2, double setVal) {
            if(current_buffer == 1){
                data0[index1 * length + index2] = setVal;
            }
            else{
                data1[index1 * length + index2] = setVal;
            }
        }

        __host__ __device__
        void switch_buffer() {
            current_buffer = !current_buffer;
        }

        __host__
        ~Grid(){
            free(data0);
            free(data1);
        }
    };

    struct UpdateResult {
        bool stable;
        double temp;
    };

    __host__ __device__
    UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state) {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1) {
            result.temp = state.border_temp;
        } else if (i == state.source_x && j == state.source_y) {
            result.temp = state.source_temp;
        } else {
            auto sum = (grid.fetch(i + 1, j) + grid.fetch(i - 1, j) + grid.fetch(i, j + 1) + grid.fetch(i, j - 1));
            switch (state.algo) {
                case Algorithm::Jacobi:
                    result.temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    result.temp = grid.fetch(i, j) + (1.0 / state.sor_constant) * (sum - 4.0 * grid.fetch(i, j));
                    break;
            }
        }
        result.stable = fabs(grid.fetch(i, j) - result.temp) < state.tolerance;
        return result;
    }

    __global__
    void cuda_calculate(const State &state, Grid &grid, int totalThreadSize, bool * stateList){
        // total data size
        int dataSize = state.room_size*state.room_size;

        bool stabilized = true;

        int taskNum = getThreadId();

        // calculate start and end position at buffer
        int startPosition = (dataSize/totalThreadSize)*taskNum + min(dataSize%totalThreadSize, taskNum);
        int endPosition = (dataSize/totalThreadSize)*(taskNum+1) + min(dataSize%totalThreadSize, taskNum+1)-1;
        int jobSize = endPosition-startPosition+1;

        // calculate start i and j
        int startI = startPosition/state.room_size;
        int i, j, count;    
        bool switchOnce = false;

        switch (state.algo) {
            case Algorithm::Jacobi:
                i = startI;
                j = startPosition%state.room_size;
                count = 0;
                for (; i < state.room_size; ++i) {
                    if(i!=startI){
                        j=0;
                    }
                    for (; j < state.room_size; ++j) {
                        // avoid extra calculation
                        if(count>=jobSize){
                            // break outside for loop
                            i = state.room_size;
                            break;
                        }
                        count++;

                        auto result = update_single(i, j, grid, state);
                        stabilized &= result.stable;
                        grid.set(alt, i, j, result.temp);
                    }
                }
                break;
            case Algorithm::Sor:
                for (auto k : {0, 1}) {
                    i = startI;
                    j = startPosition%state.room_size;
                    count = 0;
                    for (; i < state.room_size; i++) {
                        if(i!=startI){
                            j=0;
                        }
                        for (; j < state.room_size; j++) {
                            // avoid extra calculation
                            if(count>=jobSize){
                                // break outside for loop
                                i = state.room_size;
                                break;
                            }
                            count++;

                            if (k == ((i + j) & 1)) {
                                auto result = update_single(i, j, grid, state);
                                stabilized &= result.stable;
                                grid.set(alt, i, j, result.temp);
                            } else {
                                grid.set(alt, i, j, grid.fetch(i, j));
                            }
                        }
                    }

                    // make sure all threads finish calculation
                    __syncthreads();
                    if(taskNum==0 && switchOnce == false){
                            grid.switch_buffer();
                            switchOnce = true;
                    }
                    // make sure all threads finish calculation
                    __syncthreads();

                }
        }
        
        stateList[taskNum] = stabilized;
    }

    __host__
    bool calculate(const State &state, Grid &grid, int totalThreadSize) {

        // allocate state
        State * stateCopyPtr;
        cudaMallocManaged((void**)&stateCopyPtr, sizeof(state));
        cudaMemcpy(stateCopyPtr, &state, sizeof(state), cudaMemcpyHostToDevice);
        State & stateCopyReference = *stateCopyPtr;

        // allocate grid
        Grid * gridCopyPtr;
        double *data0, *data1;
        cudaMallocManaged((void**)&gridCopyPtr, sizeof(grid));
        cudaMallocManaged((void**)&data0, state.room_size*state.room_size*sizeof(double));
        cudaMallocManaged((void**)&data1, state.room_size*state.room_size*sizeof(double));
        cudaMemcpy(gridCopyPtr, &grid, sizeof(grid), cudaMemcpyHostToDevice);
        Grid & gridCopyReference = *gridCopyPtr;
        gridCopyReference.data0 = data0;
        gridCopyReference.data1 = data1;
        cudaMemcpy(gridCopyReference.get_current_buffer(), grid.get_current_buffer(), state.room_size*state.room_size*sizeof(double), cudaMemcpyHostToDevice);

        // allocate stateList
        bool * stateList;
        cudaMallocManaged((void**)&stateList, totalThreadSize*sizeof(bool));
        memset(stateList, true, sizeof(stateList));

        cuda_calculate<<<1, totalThreadSize>>>(stateCopyReference, gridCopyReference, totalThreadSize, stateList);
        cudaDeviceSynchronize();

        gridCopyReference.switch_buffer();
        grid.switch_buffer();

        // copy back the newest data
        memcpy(grid.get_current_buffer(), gridCopyReference.get_current_buffer(), state.room_size*state.room_size*sizeof(double));

        bool statelizedAll = true;

        for(int i=0;i<totalThreadSize; i++){
            // printf("checking %d\n", stabilized);
            statelizedAll &= stateList[i];
        }

        // free allocated space
        cudaFree(stateCopyPtr);
        cudaFree(gridCopyPtr);
        cudaFree(data0);
        cudaFree(data1);
        cudaFree(stateList);

        return statelizedAll;
    };

} // namespace hdist