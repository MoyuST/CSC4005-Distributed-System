#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cstring>

struct Square {
    std::vector<int> buffer;
    size_t length;

    explicit Square(size_t length) : buffer(length), length(length * length) {}

    void resize(size_t new_length) {
        buffer.assign(new_length * new_length, false);
        length = new_length;
    }

    auto& operator[](std::pair<size_t, size_t> pos) {
        return buffer[pos.second * length + pos.first];
    }
};

// struct to hold parameters
struct Arguments {
    int* buffer;
    int size;
    int scale;
    double x_center;
    double y_center;
    int k_value;
    int taskNum;
    int threadSize;
};

void *subTask(void *arg_ptr) {
    // retrieve information from passed parameters
    auto arguments = static_cast<Arguments *>(arg_ptr);
    int piecesNum = arguments->size*arguments->size/arguments->threadSize;
    int remainder = arguments->size*arguments->size%arguments->threadSize;
    int startPosi = arguments->taskNum*piecesNum+std::min(arguments->taskNum, remainder);
    int endPosi = (arguments->taskNum+1)*piecesNum+std::min(arguments->taskNum+1,remainder)-1;
    int taskSize = endPosi-startPosi+1;

    // prepare for calculation task size
    int startI = startPosi/arguments->size;
    int i = startI;
    int j = startPosi%arguments->size;
    int count = 0;

    double cx = static_cast<double>(arguments->size) / 2 + arguments->x_center;
    double cy = static_cast<double>(arguments->size) / 2 + arguments->y_center;
    double zoom_factor = static_cast<double>(arguments->size) / 4 * arguments->scale;
    for (; i < arguments->size; ++i) {
        if(i!=startI){
            j=0;
        }
        for (; j < arguments->size; ++j) {
            // if finish task then return
            if(count>=taskSize){
                return nullptr;
            }
            count++;
            double x = (static_cast<double>(i) - cx) / zoom_factor;
            double y = (static_cast<double>(j) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < arguments->k_value);
            arguments->buffer[i*arguments->size+j] = k;
        }
    }

    return nullptr;
}

void calculate(Square &buffer, int size, int scale, double x_center, double y_center, int k_value, int threadSize) {
    // fork threads
    std::vector<pthread_t> threads(threadSize);

    // initialize each thread with task information
    for(int i=0;i<threadSize;i++){
        pthread_create(&threads[i], nullptr, subTask, new Arguments{
                    .buffer = buffer.buffer.data(),
                    .size = size,
                    .scale = scale,
                    .x_center = x_center,
                    .y_center = y_center,
                    .k_value = k_value,
                    .taskNum = i,
                    .threadSize = threadSize
        });
    }

    // wait until all the threads finish
    for (auto & i : threads) {
        pthread_join(i, nullptr);
    }
}

int main(int argc, char **argv) {
    // use 1 threadsd as default
    int threadSize = 1;

    // fetch thread size from command line
    if(argc>=2){
        threadSize = atoi(argv[1]);
    }

    // avoid invalid input
    if(threadSize==0){
        threadSize = 1;
    }

    std::cout << "testing with " << threadSize << " thread(s)" << std::endl;

    int testSize[5] = {1000, 2000, 4000, 8000, 16000};

    Square canvas(100);
    size_t duration = 0;
    static int center_x = 0;
    static int center_y = 0;
    static int size = 800;
    static int scale = 1;
    static int k_value = 100;
    {
        using namespace std::chrono;
        for(int i=0;i<5;i++){
            duration = 0;
            size = testSize[i];
            canvas.resize(size);
            auto begin = high_resolution_clock::now();
            calculate(canvas, size, scale, center_x, center_y, k_value, threadSize);
            auto end = high_resolution_clock::now();
            duration += duration_cast<nanoseconds>(end - begin).count();
            std::cout << "plot size: " << testSize[i] <<
            " thread size: " << threadSize << " time spent: " << duration << " ns" << std::endl;
        }
    }

    return 0;
}
