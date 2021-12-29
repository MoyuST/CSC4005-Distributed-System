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

            // local buffer holding data sent from other process
            Element data_from_other_process;

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
            
            // allocate buffer holding data
            std::vector<Element> local_data_buffer(data_package_size);

            // scatter data
            MPI_Scatter(padded_data_array.data(), data_package_size, MPI_LONG, local_data_buffer.data(), data_package_size, MPI_LONG, 0, MPI_COMM_WORLD);

            // step 2
            // sort data until sorted
            bool sorted_flag_all = false;
            bool sorted_flag_local = false;

            while(!sorted_flag_all){
                sorted_flag_local = true;

                // sort the odd position
                for(int i=1;i<data_package_size-1;i+=2){
                    if(local_data_buffer[i] > local_data_buffer[i+1]){
                        Element temp = local_data_buffer[i];
                        local_data_buffer[i] = local_data_buffer[i+1];
                        local_data_buffer[i+1] = temp;
                        sorted_flag_local = false;
                    }
                }

                // odd rank send data
                if(size % 2 != 0 && rank == size-1){} // last process would do nothing if the total size is odd
                else if(rank % 2 != 0){
                    // send first number to previous process and wait for result
                    MPI_Sendrecv(local_data_buffer.data(), 1, MPI_LONG, (rank-1+size)%size, 0,
                                local_data_buffer.data(), 1, MPI_LONG, (rank-1+size)%size, 0, 
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else{
                    // receive data from other process and compare the value with the last number
                    MPI_Recv(&data_from_other_process, 1, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data_from_other_process < local_data_buffer[data_package_size-1]){
                        Element temp = data_from_other_process;
                        data_from_other_process = local_data_buffer[data_package_size-1];
                        local_data_buffer[data_package_size-1] = temp;
                        sorted_flag_local = false;
                    }
                    MPI_Send(&data_from_other_process, 1, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD);
                }

                // sort the even position
                for(int i=0;i<data_package_size-1;i+=2){
                    if(local_data_buffer[i] > local_data_buffer[i+1]){
                        Element temp = local_data_buffer[i];
                        local_data_buffer[i] = local_data_buffer[i+1];
                        local_data_buffer[i+1] = temp;
                        sorted_flag_local = false;
                    }
                }

                // even rank send data
                if(rank != 0){
                    if(size%2 == 0 && rank == size-1){} // last process would do nothing if the total size is even
                    else if(rank % 2 != 1){
                        // send first number to previous process and wait for result
                        MPI_Sendrecv(local_data_buffer.data(), 1, MPI_LONG, (rank-1+size)%size, 0,
                                    local_data_buffer.data(), 1, MPI_LONG, (rank-1+size)%size, 0, 
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    else{
                        // receive data from other process and compare the value with the last number
                        MPI_Recv(&data_from_other_process, 1, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if(data_from_other_process < local_data_buffer[data_package_size-1]){
                            Element temp = data_from_other_process;
                            data_from_other_process = local_data_buffer[data_package_size-1];
                            local_data_buffer[data_package_size-1] = temp;
                            sorted_flag_local = false;
                        }
                        MPI_Send(&data_from_other_process, 1, MPI_LONG, (rank+1+size)%size, 0, MPI_COMM_WORLD);
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
