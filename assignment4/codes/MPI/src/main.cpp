#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>

template<typename ...Args>
void UNUSED(Args &&... args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

static MPI_Datatype MPI_State;

int main(int argc, char **argv) {
    int rank;
    int processSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processSize);

    // initialize MPI struct
    const int nitems=9;
    int blocklengths[9] = {1,1,1,1,1,1,1,1,1};
    MPI_Datatype types[9] = {MPI_INT, MPI_FLOAT, MPI_INT, MPI_INT, 
                MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};

    MPI_Aint offsets[9];

    offsets[0] = offsetof(hdist::State, room_size);
    offsets[1] = offsetof(hdist::State, block_size);
    offsets[2] = offsetof(hdist::State, source_x);
    offsets[3] = offsetof(hdist::State, source_y);
    offsets[4] = offsetof(hdist::State, source_temp);
    offsets[5] = offsetof(hdist::State, border_temp);
    offsets[6] = offsetof(hdist::State, tolerance);
    offsets[7] = offsetof(hdist::State, sor_constant);
    offsets[8] = offsetof(hdist::State, algo);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_State);
    MPI_Type_commit(&MPI_State);


    UNUSED(argc, argv);
    bool first = true;
    bool finished = false;
    static hdist::State current_state, last_state;
    auto grid = hdist::Grid{
            static_cast<size_t>(current_state.room_size),
            current_state.border_temp,
            current_state.source_temp,
            static_cast<size_t>(current_state.source_x),
            static_cast<size_t>(current_state.source_y)};


    if(rank == 0){
        printf("MPI: launch %d processes\n", processSize);

        // test codes
        // int testCases[] = {200, 400, 800, 1600, 3200};
        // int testCaseIdx = 0, testCnt = 0;
        // static std::chrono::high_resolution_clock::time_point beginIteration, endIteration;

        static std::chrono::high_resolution_clock::time_point begin, end;
        static const char* algo_list[2] = { "jacobi", "sor" };
        graphic::GraphicContext context{"Assignment 4"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
            auto io = ImGui::GetIO();
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::Begin("Assignment 4", nullptr,
                        ImGuiWindowFlags_NoMove
                        | ImGuiWindowFlags_NoCollapse
                        | ImGuiWindowFlags_NoTitleBar
                        | ImGuiWindowFlags_NoResize);
            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
            ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
            ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
            ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
            ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
            ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

            // testCnt++;

            // if(testCnt == 5){
            //     current_state.room_size = testCases[testCaseIdx];
            //     testCaseIdx = (testCaseIdx+1) % 5;
            //     testCnt = 0;
            // }

            if (current_state.algo == hdist::Algorithm::Sor) {
                ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
            }

            // beginIteration = std::chrono::high_resolution_clock::now();

            // broadcast current state
            MPI_Bcast(&current_state, 1, MPI_State, 0, MPI_COMM_WORLD);

            // change size regenerate
            if (current_state.room_size != last_state.room_size) {
                grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y)};
                first = true;
            }

            if (current_state != last_state) {
                last_state = current_state;
                finished = false;
            }

            if (first) {
                first = false;
                finished = false;
                begin = std::chrono::high_resolution_clock::now();
            }

            if (!finished) {
                int dataSize = current_state.room_size*current_state.room_size;

                // bcast get_current_buffer
                MPI_Bcast(grid.get_current_buffer().data(), dataSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // do calculation
                finished = hdist::calculate(current_state, grid);

                bool localFinished = finished;

                // reduce and broadcast finished
                MPI_Reduce(&localFinished, &finished, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
                MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

                // gatherv get_current_buffer result from all ranks
                // calculate the job size of each rank
                std::vector<int> taskSizeList;
                for(int i=0;i<processSize;i++){
                    int startPosition = (dataSize/processSize)*i + std::min(dataSize%processSize, i);
                    int endPosition = (dataSize/processSize)*(i+1) + std::min(dataSize%processSize, i+1)-1;
                    taskSizeList.push_back(endPosition-startPosition+1);
                }

                // calculate the strips of received data packages
                std::vector<int> strips;
                strips.push_back(0);
                for(int i=1;i<processSize;i++){
                    strips.push_back(strips[i-1]+taskSizeList[i-1]);
                }

                // calcualte job size of itself
                int startPosition = (dataSize/processSize)*rank + std::min(dataSize%processSize, rank);
                int endPosition = (dataSize/processSize)*(rank+1) + std::min(dataSize%processSize, rank+1)-1;
                int jobSize = endPosition-startPosition+1;

                // rank 0 gather calculated result
                MPI_Gatherv(grid.get_current_buffer().data()+startPosition, jobSize, MPI_DOUBLE, 
                    grid.get_current_buffer().data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if (finished) end = std::chrono::high_resolution_clock::now();

                // endIteration = std::chrono::high_resolution_clock::now();
                // printf("processes %d\tone iteration %lld (ns)\troom size %d\n", 
                // processSize, std::chrono::duration_cast<std::chrono::nanoseconds>(endIteration - beginIteration).count(), current_state.room_size);

            } else {
                ImGui::Text("stabilized in %lld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
                printf("stabilized in %lld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            }

            const ImVec2 p = ImGui::GetCursorScreenPos();
            float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
            for (size_t i = 0; i < current_state.room_size; ++i) {
                for (size_t j = 0; j < current_state.room_size; ++j) {
                    auto temp = grid[{i, j}];
                    auto color = temp_to_color(temp);
                    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                    y += current_state.block_size;
                }
                x += current_state.block_size;
                y = p.y + current_state.block_size;
            }
            ImGui::End();
        });
    }
    else{
        while(true){
            // receive current state
            MPI_Bcast(&current_state, 1, MPI_State, 0, MPI_COMM_WORLD);

            // change size regenerate
            if (current_state.room_size != last_state.room_size) {
                grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y)};
                first = true;
            }

            if (current_state != last_state) {
                last_state = current_state;
                finished = false;
            }

            if (first) {
                first = false;
                finished = false;
            }

            if (!finished) {
                int dataSize = current_state.room_size*current_state.room_size;

                // bcast get_current_buffer
                MPI_Bcast(grid.get_current_buffer().data(), dataSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // do calculation
                finished = hdist::calculate(current_state, grid);

                // reduce and receive finished
                MPI_Reduce(&finished, nullptr, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
                MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

                // send get_current_buffer result to rank 0
                // calcualte job size of itself
                int startPosition = (dataSize/processSize)*rank + std::min(dataSize%processSize, rank);
                int endPosition = (dataSize/processSize)*(rank+1) + std::min(dataSize%processSize, rank+1)-1;
                int jobSize = endPosition-startPosition+1;

                // send calculated result
                MPI_Gatherv(grid.get_current_buffer().data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, 
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

        }
    }

    // if ploting windows closed, rank 0 would then stop all other processes
    if(rank==0){
        printf("aborting all processes\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Type_free(&MPI_State);
    MPI_Finalize();
}
