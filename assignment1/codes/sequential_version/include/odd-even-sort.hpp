#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <chrono>
#include <vector>

namespace sort {
    using Element = int64_t;
    /** Information
     *  The information for a single mpi run
     */
    struct Information {
        std::chrono::high_resolution_clock::time_point start{};
        std::chrono::high_resolution_clock::time_point end{};
        size_t length{};
        int num_of_proc{};
        int argc{};
        std::vector<char *> argv{};
    };

    struct Context {
        int argc;
        char **argv;

        Context(int &argc, char **&argv);

        ~Context();

        /**!
         * Sort the elements in range [begin, end) in the ascending order.
         * For sub-processes, null pointers will be passed. That is, the root process
         * should be in charge of sending the data to other processes.
         * @param begin starting position
         * @param end ending position
         * @return the information for the sorting
         */
        std::unique_ptr<Information> mpi_sort(Element *begin, Element *end) const;

        /*!
         * Print out the information.
         * @param info information struct
         * @param output output stream
         * @return the output stream
         */
        static std::ostream &print_information(const Information &info, std::ostream &output);
    };
}