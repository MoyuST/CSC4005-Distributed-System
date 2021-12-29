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

        std::unique_ptr<Information> information{};

        res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);


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
            if(rank == 0){
                bool unchanged_flag = false;
                int array_size = end - begin;
                while(!unchanged_flag){
                    unchanged_flag = true;
                    // odd position
                    for(int i=1;i<array_size-1;i+=2){
                        if(begin[i] > begin[i+1]){
                            Element temp = begin[i];
                            begin[i] = begin[i+1];
                            begin[i+1] = temp;
                            unchanged_flag = false;
                        }
                    }

                    // even position
                    for(int i=0;i<array_size-1;i+=2){
                        if(begin[i] > begin[i+1]){
                            Element temp = begin[i];
                            begin[i] = begin[i+1];
                            begin[i+1] = temp;
                            unchanged_flag = false;
                        }
                    }
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
