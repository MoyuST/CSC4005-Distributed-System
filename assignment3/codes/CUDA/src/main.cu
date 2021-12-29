#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

int atoiCustomized(char ** argv, int argc, int argvIndex){
    int result = 1;

    if(argc>argvIndex){
        result = atoi(argv[argvIndex]);
    }

    // avoid invalid input
    if(result==0){
        result = 1;
    }

    return result;
}

int main(int argc, char **argv) {
    dim3 grid_size;
    dim3 block_size;

    // use 1 as default value
    grid_size.x = atoiCustomized(argv, argc, 1);
    grid_size.y = atoiCustomized(argv, argc, 2);
    grid_size.z = atoiCustomized(argv, argc, 3);
    block_size.x = atoiCustomized(argv, argc, 4);
    block_size.y = atoiCustomized(argv, argc, 5);
    block_size.z = atoiCustomized(argv, argc, 6);

    // cudaThreadSetLimit(cudaLimitMallocHeapSize, 10*100*sizeof(double));

    printf("dim: %d %d %d  block: %d %d %d\n", grid_size.x, grid_size.y, grid_size.z, 
            block_size.x, block_size.y, block_size.z);

    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 200;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    static bool nanOccur = false;

    BodyPool * pool = new BodyPool(static_cast<size_t>(bodies), space, max_mass);

    graphic::GraphicContext context{"Assignment 3"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 3", nullptr,
                    ImGuiWindowFlags_NoMove
                    | ImGuiWindowFlags_NoCollapse
                    | ImGuiWindowFlags_NoTitleBar
                    | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);

        // reconstruct the map when necessary
        if (nanOccur == true || current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            free(pool);
            pool = new BodyPool(static_cast<size_t>(bodies), space, max_mass);
            nanOccur = false;
        }

        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            
            // record calculation time
            auto begin = std::chrono::high_resolution_clock::now();

            // calculate information for each body
            pool->update_for_tick(elapse, gravity, space, radius, grid_size, block_size, pool);

            // record calculation time
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            std::cout << pool->size() << " bodies in last " << duration << " nanoseconds" << std::endl;

            // plot calculation results
            for (int i = 0; i < pool->size(); ++i) {
                auto body = pool->get_body(i);
                if(isnan(body.get_x()) || isnan(body.get_y())){
                    nanOccur = true;
                    printf("nan occured bodypool reset\n");
                    break;
                }
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }

        }
        ImGui::End();
    });

}
