#pragma once

#include <vector>
#include <iostream>

namespace hdist {
    // use to hold the barrier
    static pthread_barrier_t barrier;

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

        static void switch_buffer(Grid & grid) {
            grid.current_buffer = !grid.current_buffer;
        }
    };

    // struct to hold parameters
    struct Arguments{
        const State &state;
        Grid &grid;
        std::vector<int> &stateList;
        int taskNum;
        int totalThreadSize;
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

    void* calculateInside(void *arg_ptr){
        // retrieve information from passed parameters
        auto arguments = static_cast<Arguments *>(arg_ptr);
        const State & state = arguments->state;
        Grid & grid = arguments->grid;
        std::vector<int> & stateList = arguments->stateList;
        int taskNum = arguments->taskNum;
        int totalThreadSize = arguments->totalThreadSize;

        // std::cout << &state << " " << &grid << " " << &stateList << " [" << taskNum << std::endl;

        bool stabilized = true;

        int dataSize = state.room_size*state.room_size;

        // calculate start and end position at buffer
        int startPosition = (dataSize/totalThreadSize)*taskNum + std::min(dataSize%totalThreadSize, taskNum);
        int endPosition = (dataSize/totalThreadSize)*(taskNum+1) + std::min(dataSize%totalThreadSize, taskNum+1)-1;
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
                        grid[{alt, i, j}] = result.temp;
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
                                grid[{alt, i, j}] = result.temp;
                            } else {
                                grid[{alt, i, j}] = grid[{i, j}];
                            }
                        }
                    }
                    // wait until all threads finish calculation
                    pthread_barrier_wait(&barrier);
                    if(taskNum==0 && switchOnce==false){
                        grid.switch_buffer();
                        switchOnce = true;
                    }
                    pthread_barrier_wait(&barrier);
                }
        }

        stateList[taskNum] = stabilized;

        return nullptr;
    }

    bool calculate(const State &state, Grid &grid, int totalThreadSize) {
        std::vector<int> stateList(totalThreadSize, true);

        std::vector<pthread_t> threads(totalThreadSize);

        pthread_barrier_init(&barrier, NULL, totalThreadSize);

        // std::cout << &state << " " << &grid << " " << &stateList << std::endl;

        // initialize each thread with task information
        for(int i=0;i<totalThreadSize;i++){
            pthread_create(&threads[i], nullptr, calculateInside, new Arguments{
                    .state = state,
                    .grid = grid,
                    .stateList = stateList,
                    .taskNum = i,
                    .totalThreadSize = totalThreadSize
            });
        }

        // wait until all the threads finish
        for (auto & i : threads) {
            pthread_join(i, nullptr);
        }

        pthread_barrier_destroy(&barrier);

        bool statelizedAll = true;

        for(bool stabilized : stateList){
            // printf("checking %d\n", stabilized);
            statelizedAll &= stabilized;
        }

        grid.switch_buffer();

        return statelizedAll;
    }



} // namespace hdist
