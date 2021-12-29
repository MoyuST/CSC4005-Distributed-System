#pragma once

#include <vector>
#include <mpi.h>
#include <omp.h>

namespace hdist {

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

        bool operator==(const State &that) const = default;
    };

    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        std::vector<double> data0, data1;
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                : data0(size * size), data1(size * size), length(size) {
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        std::vector<double> &get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        std::vector<double> &get_alternate_buffer() {
            if (current_buffer == 0) return data1;
            return data0;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[
                    std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer() {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult {
        bool stable;
        double temp;
    };

    UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state) {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1) {
            result.temp = state.border_temp;
        } else if (i == state.source_x && j == state.source_y) {
            result.temp = state.source_temp;
        } else {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            switch (state.algo) {
                case Algorithm::Jacobi:
                    result.temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    result.temp = grid[{i, j}] + (1.0 / state.sor_constant) * (sum - 4.0 * grid[{i, j}]);
                    break;
            }
        }
        result.stable = fabs(grid[{i, j}] - result.temp) < state.tolerance;
        return result;
    }

    bool calculate(const State &state, Grid &grid, int totalThreadSize) {
        bool stabilized = true;
        int rank;
        int processSize;
        std::vector<int> stateList(totalThreadSize, true);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processSize);

        int dataSize = state.room_size*state.room_size;

        // calculate start and end position at buffer
        int processStartPosition = (dataSize/processSize)*rank + std::min(dataSize%processSize, rank);
        int processEndPosition = (dataSize/processSize)*(rank+1) + std::min(dataSize%processSize, rank+1)-1;
        int processTaskSize = processEndPosition - processStartPosition + 1;

        #pragma omp parallel
        {
            int taskNum = omp_get_thread_num();

            int startPosition = (processTaskSize/totalThreadSize)*taskNum + std::min(processTaskSize%totalThreadSize, taskNum) + processStartPosition;
            int endPosition = (processTaskSize/totalThreadSize)*(taskNum+1) + std::min(processTaskSize%totalThreadSize, taskNum+1)-1 + processStartPosition;
            int jobSize = endPosition-startPosition+1;

            // calculate start i and j
            int startI = startPosition/state.room_size;
            int i, j, count, endI;    
            bool switchOnce = false;
            int starti, startj, curi;

            switch (state.algo) {
                case Algorithm::Jacobi:
                    i = startI;
                    j = startPosition%state.room_size;
                    starti = i;
                    startj = j;
                    count = 0;
                    for (; i < state.room_size; ++i) {
                        if(i!=startI){
                            j=0;
                        }
                        curi = i;
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
                            grid[{alt, i, j}] = result.temp;
                        }
                    }
                    break;
                case Algorithm::Sor:
                    for (auto k : {0, 1}) {
                        i = startI;
                        j = startPosition%state.room_size;
                        count = 0;

                        // calculate the left and top extra points
                        if(switchOnce==false){
                            // left points
                            if(i!=0){
                                int x = i-1;
                                for(int y=j;y<state.room_size;y++){
                                    if (k == ((x + y) & 1)) {
                                        auto result = update_single(x, y, grid, state);
                                        // stabilized &= result.stable;
                                        grid[{alt, x, y}] = result.temp;
                                    } else {
                                        grid[{alt, x, y}] = grid[{x, y}];
                                    }
                                }
                            }

                        // top points
                        for(int y=0; y<j;y++){
                            if (k == ((i + y) & 1)) {
                                auto result = update_single(i, y, grid, state);
                                // stabilized &= result.stable;
                                grid[{alt, i, y}] = result.temp;
                            } else {
                                grid[{alt, i, y}] = grid[{i, y}];
                            }
                        }
                    }

                    for (; i < state.room_size; i++) {
                        if(i!=startI){
                            j=0;
                        }

                        for (; j < state.room_size; j++) {
                            // avoid extra calculation
                            if(count>=jobSize){
                                // break outside for loop
                                endI = i;
                                i = state.room_size;
                                break;
                            }
                            count++;

                            if (k == ((i + j) & 1)) {
                                auto result = update_single(i, j, grid, state);
                                stabilized &= result.stable;
                                grid[{alt, i, j}] = result.temp;
                            } else {
                                grid[{alt, i, j}] = grid[{i, j}];
                            }
                        }
                    }

                    // calculate the right and bottom extra points
                    if(switchOnce==false){
                        // last point reach the bottom exactly
                        int rightJ = j;
                        int rightI = endI;
                        if(j==0){
                            rightJ = state.room_size-1;
                            // block bottom points calculation
                            j = state.room_size;
                        }
                        else{
                            rightI++;
                        }

                        // calculate right points
                        if(rightI<state.room_size){
                            int x = rightI;

                            for(int y=0;y<rightJ; y++){
                                if (k == ((x + y) & 1)) {
                                    auto result = update_single(x, y, grid, state);
                                    // stabilized &= result.stable;
                                    grid[{alt, x, y}] = result.temp;
                                } else {
                                    grid[{alt, x, y}] = grid[{x, y}];
                                }
                            }
                        }

                        // calculate bottom points
                        for(int y=j;y<state.room_size;y++){
                            if (k == ((endI + y) & 1)) {
                                auto result = update_single(endI, y, grid, state);
                                // stabilized &= result.stable;
                                grid[{alt, endI, y}] = result.temp;
                            } else {
                                grid[{alt, endI, y}] = grid[{endI, y}];
                            }
                        }

                    }

                    // wait until all threads finish calculation
                    #pragma omp barrier

                    if(taskNum==0 && switchOnce == false){
                        grid.switch_buffer();
                    }

                    switchOnce = true;
                    
                    #pragma omp barrier

                }
            }

            stateList[taskNum] = stabilized;

            // printf("%d %d %d %d %d %d [%d]\n", rank, taskNum, starti, startj, curi, j, stabilized);

        }

        bool statelizedAll = true;

        for(bool stabilized : stateList){
            // printf("checking %d\n", stabilized);
            statelizedAll &= stabilized;
        }

        grid.switch_buffer();

        return statelizedAll;
    };


} // namespace hdist
