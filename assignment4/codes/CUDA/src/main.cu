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

int main(int argc, char **argv) {
    // use 4 threadsd as default
    int totalThreadSize = 4;

    // fetch thread size from command line
    if(argc>=2){
        totalThreadSize = atoi(argv[1]);
    }

    // avoid invalid input
    if(totalThreadSize==0){
        totalThreadSize = 4;
    }

    printf("CUDA: launch %d threads\n", totalThreadSize);
    // test codes
    // int testCases[] = {200, 400, 800, 1600, 3200};
    // int testCaseIdx = 0, testCnt = 0;
    // static std::chrono::high_resolution_clock::time_point beginIteration, endIteration;

    UNUSED(argc, argv);
    bool first = true;
    bool finished = false;
    static hdist::State current_state, last_state;
    static std::chrono::high_resolution_clock::time_point begin, end;
    static const char* algo_list[2] = { "jacobi", "sor" };
    graphic::GraphicContext context{"Assignment 4"};
    hdist::Grid * gridPtr = new hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};
    hdist::Grid & grid = *gridPtr;
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

        if (current_state.room_size != last_state.room_size) {
            delete gridPtr;
            gridPtr = new hdist::Grid{
                static_cast<size_t>(current_state.room_size),
                current_state.border_temp,
                current_state.source_temp,
                static_cast<size_t>(current_state.source_x),
                static_cast<size_t>(current_state.source_y)};
            grid = *gridPtr;
            first = true;
        }

        if (!(current_state == last_state)) {
            last_state = current_state;
            finished = false;
        }

        if (first) {
            first = false;
            finished = false;
            begin = std::chrono::high_resolution_clock::now();
        }

        if (!finished) {
            finished = hdist::calculate(current_state, grid, totalThreadSize);
            if (finished) end = std::chrono::high_resolution_clock::now();

            // endIteration = std::chrono::high_resolution_clock::now();
            // printf("thread %d\tone iteration %lld (ns)\troom size %d\n", 
            // totalThreadSize, std::chrono::duration_cast<std::chrono::nanoseconds>(endIteration - beginIteration).count(), current_state.room_size);
        } else {
            ImGui::Text("stabilized in %lld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            printf("stabilized in %lld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        }

        const ImVec2 p = ImGui::GetCursorScreenPos();
        float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
        for (size_t i = 0; i < current_state.room_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {
                auto temp = grid.fetch(i, j);
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
