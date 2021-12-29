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

    explicit Square(size_t length) : buffer(length*length), length(length) {}

    void resize(size_t new_length) {
        buffer.assign(new_length * new_length, false);
        length = new_length;
    }

    auto& operator[](std::pair<size_t, size_t> pos) {
        return buffer[pos.second * length + pos.first];
    }
};

// define struct to pass information
typedef struct Arguments_S {
        int size;
        int scale;
        int k_value;
        double x_center;
        double y_center;
} Arguments;

static MPI_Datatype Arguments_TYPE;

void calculate(Square &buffer, int size, int rank, int processSize, int scale, double x_center, double y_center, int k_value) {
    // rank 0 broadcast parameters
    Arguments arguments;

    if(rank==0){
        arguments.size = size;
        arguments.scale = scale;
        arguments.k_value = k_value;
        arguments.x_center = x_center;
        arguments.y_center = y_center;
    }

    MPI_Bcast(&arguments, 1, Arguments_TYPE, 0, MPI_COMM_WORLD);

    // retreive data
    if(rank!=0){
        size = arguments.size;
        scale = arguments.scale;
        k_value = arguments.k_value;
        x_center = arguments.x_center;
        y_center = arguments.y_center;
    }

    // allcate buffer
    std::vector<int> calculationResult;

    int piecesNum = size*size/processSize;
    int remainder = size*size%processSize;
    int startPosi = rank*piecesNum+std::min(rank, remainder);
    int endPosi = (rank+1)*piecesNum+std::min(rank+1,remainder)-1;
    int taskSize = endPosi-startPosi+1;

    // prepare for calculation task size
    int startI = startPosi/size;
    int i = startI;
    int j = startPosi%size;
    int count = 0;

    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    for (; i < size; ++i) {
        if(i!=startI){
            j=0;
        }
        for (; j < size; ++j) {
            // avoid extra calculation
            if(count>=taskSize){
                // break outside for loop
                i = size;
                break;
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
            } while (norm(z) < 2.0 && k < k_value);
            calculationResult.push_back(k);
        }
    }

    // gather calculation result
    // first gather the sizes of ech subtask
    std::vector<int> taskSizeList;
    if(rank==0){ 
        taskSizeList.resize(processSize);
    }

    MPI_Gather(&taskSize, 1, MPI_INT, taskSizeList.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> strips;
    // rank 0 calculate strips of received data packages
    if(rank==0){
    	strips.push_back(0);
        for(int i=1;i<processSize;i++){
            strips.push_back(strips[i-1]+taskSizeList[i-1]);
        }
    }

    MPI_Gatherv(calculationResult.data(), taskSize, MPI_INT, buffer.buffer.data(), taskSizeList.data(), strips.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    int rank;
    int processSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processSize);

    // initialize MPI struct
    const int nitems=5;
    int          blocklengths[5] = {1,1,1,2,2};
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint     offsets[5];

    offsets[0] = offsetof(Arguments, size);
    offsets[1] = offsetof(Arguments, scale);
    offsets[2] = offsetof(Arguments, k_value);
    offsets[3] = offsetof(Arguments, x_center);
    offsets[4] = offsetof(Arguments, y_center);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &Arguments_TYPE);
    MPI_Type_commit(&Arguments_TYPE);

    // holding the test size
    int testSize[5] = {1000, 2000, 4000, 8000, 16000};

    for(int i=0;i<5;i++){
        if (0 == rank) {
            Square canvas(0);
            size_t duration = 0;
            size_t pixels = 0;
            static int center_x = 0;
            static int center_y = 0;
            int size = testSize[i];
            static int scale = 1;
            static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
            static int k_value = 100;
            {
                using namespace std::chrono;
                canvas.resize(size);
                // rank 0 start timer
                auto begin = high_resolution_clock::now();
                calculate(canvas, size, rank, processSize, scale, center_x, center_y, k_value);
                auto end = high_resolution_clock::now();
                pixels += size;
                duration += duration_cast<nanoseconds>(end - begin).count();
                // rank 0 report result for one iteration
                std::cout << "plot size: " << size << 
                " process size: " << processSize << " time spent: " << duration << " ns" << std::endl;
            }
        }
        else{
            Square canvas(0);
            calculate(canvas, 0, rank, processSize, 0, 0, 0, 0);
        }
    }


    MPI_Type_free(&Arguments_TYPE);
    MPI_Finalize();
    return 0;
}
