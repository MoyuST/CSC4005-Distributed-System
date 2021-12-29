#include <odd-even-sort.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

namespace sort {
    using namespace std::chrono;


    Context::Context(int &argc, char **&argv) : argc(argc), argv(argv) {
        MPI_Init(&argc, &argv);
    }

    Context::~Context() {
        MPI_Finalize();
    }

    std::unique_ptr<Information> Context::mpi_sort(Element *begin, Element *end) const {
        int res;
        int rank;
        int size;
        int data_package_size;
        std::vector<Element> padded_data_array;

        std::unique_ptr<Information> information{};

        res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);


        if (MPI_SUCCESS != res) {
            throw std::runtime_error("failed to get MPI world rank");
        }

        if (0 == rank) {
            information = std::make_unique<Information>();
            information->length = end - begin;
            res = MPI_Comm_size(MPI_COMM_WORLD, &information->num_of_proc);
            if (MPI_SUCCESS != res) {
                throw std::runtime_error("failed to get MPI world size");
            };
            information->argc = argc;
            for (auto i = 0; i < argc; ++i) {
                information->argv.push_back(argv[i]);
            }
            information->start = high_resolution_clock::now();
        }

        {
            /// now starts the main sorting procedure

            // step 0
            // check if size is 1
            if(size == 1){
                std::sort(begin, end);
            }
            else{
                // step 1
                // rank 0 scatter data to all process
                if (rank == 0) {
                    int original_total_data_size = end - begin;
                    int padded_total_data_size = original_total_data_size;

                    // allocate a new array for padded array
                    padded_data_array.assign(begin, end);

                    // pad positive infinity to the data array
                    int size_remainder = original_total_data_size % size;
                    if(size_remainder != 0){
                        int padded_size = size - size_remainder;
                        padded_total_data_size += padded_size;

                        for(int i=0 ; i<padded_size; i++){
                            padded_data_array.push_back(std::numeric_limits<Element>::max());
                        }
                    }
                    data_package_size = padded_total_data_size / size;
                }

                // boardcast the size of data received for each process
                MPI_Bcast(&data_package_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                
                // allocate buffer holding data and the data from other process
                std::vector<Element> local_data_buffer(data_package_size * 2);

                // scatter data
                MPI_Scatter(padded_data_array.data(), data_package_size, MPI_LONG, local_data_buffer.data(), data_package_size, MPI_LONG, 0, MPI_COMM_WORLD);

                // step 2
                // sort data until sorted
                bool sorted_flag_all = false;
                bool sorted_flag_local = false;

                while(!sorted_flag_all){
                    // odd rank send data
                    if(size % 2 != 0 && rank == size-1){
                        // last process would do nothing if the total size is odd
                        sorted_flag_local = true;
                    }
                    else if(rank % 2 != 0){
                        // send local data to previous process and wait for result
                        MPI_Sendrecv(local_data_buffer.data(), data_package_size, MPI_LONG, (rank-1+size)%size, 0,
                                    local_data_buffer.data(), data_package_size, MPI_LONG, (rank-1+size)%size, 0, 
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    else{
                        // receive data from other process
                        MPI_Recv(local_data_buffer.data()+data_package_size, data_package_size, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        sorted_flag_local = std::is_sorted(local_data_buffer.data(), local_data_buffer.data()+data_package_size*2);
                        // check whether array is already sort which means no swap will happen
                        if(!sorted_flag_local){
                            std::sort(local_data_buffer.data(), local_data_buffer.data()+data_package_size*2);
                        }
                        // return the latter half of sorted result
                        MPI_Send(local_data_buffer.data()+data_package_size, data_package_size, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD);
                    }

                    // even rank send data
                    if(rank != 0){
                        if(size%2 == 0 && rank == size-1){
                            // last process would do nothing if the total size is even
                            sorted_flag_local = true;
                        }
                        else if(rank % 2 != 1){
                            // send local data to previous process and wait for result
                            MPI_Sendrecv(local_data_buffer.data(), data_package_size, MPI_LONG, (rank-1+size)%size, 0,
                                        local_data_buffer.data(), data_package_size, MPI_LONG, (rank-1+size)%size, 0, 
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                        else{
                            // receive data from other process
                            MPI_Recv(local_data_buffer.data()+data_package_size, data_package_size, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            sorted_flag_local = std::is_sorted(local_data_buffer.data(), local_data_buffer.data()+data_package_size*2);
                            // check whether array is already sort which means no swap will happen
                            if(!sorted_flag_local){
                                std::sort(local_data_buffer.data(), local_data_buffer.data()+data_package_size*2);
                            }
                            // return the latter half of sorted result
                            MPI_Send(local_data_buffer.data()+data_package_size, data_package_size, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD);
                        }
                    }
                    
                    // check whether no swap has happened in all processes
                    MPI_Reduce(&sorted_flag_local, &sorted_flag_all, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);

                    // broadcast the signal whether another sort iteration is necessary
                    MPI_Bcast(&sorted_flag_all, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                }

                // step 3
                // gather sorted array 
                MPI_Gather(local_data_buffer.data(), data_package_size, MPI_LONG, padded_data_array.data(), data_package_size, MPI_LONG, 0, MPI_COMM_WORLD);
                if(rank == 0){
                    // strip the padded data and put needed data back to original array
                    copy(padded_data_array.begin(), padded_data_array.begin()+(end-begin), begin);
                }
            }

        }

        if (0 == rank) {
            information->end = high_resolution_clock::now();
        }
        return information;
    }

    std::ostream &Context::print_information(const Information &info, std::ostream &output) {
        auto duration = info.end - info.start;
        auto duration_count = duration_cast<nanoseconds>(duration).count();
        auto mem_size = static_cast<double>(info.length) * sizeof(Element) / 1024.0 / 1024.0 / 1024.0;
        output << "input size: " << info.length << std::endl;
        output << "proc number: " << info.num_of_proc << std::endl;
        output << "duration (ns): " << duration_count << std::endl;
        output << "throughput (gb/s): " << mem_size / static_cast<double>(duration_count) * 1'000'000'000.0
               << std::endl;
        return output;
    }
}
