#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>
#include <algorithm>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

// define struct to pass information
typedef struct Arguments_S {
        int current_bodies;
        float current_space;
        float max_mass;
        float elapse;
        float gravity;
        float radius;
} Arguments;

static MPI_Datatype Arguments_TYPE;

int main(int argc, char **argv) {
    int rank;
    int processSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processSize);

    // use 4 threads for OpenMP as default
    int totalThreadSize = 4;

    // fetch thread size from command line
    if(argc>=2){
        totalThreadSize = atoi(argv[1]);
    }

    // avoid invalid input
    if(totalThreadSize==0){
        totalThreadSize = 4;
    }

    printf("OpenMP thread size %d\n", totalThreadSize);
    omp_set_num_threads(totalThreadSize);

    // initialize MPI struct
    const int nitems=6;
    int blocklengths[6] = {1,1,1,1,1,1};
    MPI_Datatype types[6] = {MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint     offsets[6];

    offsets[0] = offsetof(Arguments, current_bodies);
    offsets[1] = offsetof(Arguments, current_space);
    offsets[2] = offsetof(Arguments, max_mass);
    offsets[3] = offsetof(Arguments, elapse);
    offsets[4] = offsetof(Arguments, gravity);
    offsets[5] = offsetof(Arguments, radius);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &Arguments_TYPE);
    MPI_Type_commit(&Arguments_TYPE);

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
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

    if(rank==0){
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

            // boardcast updated plotting information to other ranks
            Arguments arguments;
            arguments.current_bodies = current_bodies;
            arguments.current_space = current_space;
            arguments.max_mass = max_mass;
            arguments.elapse = elapse;
            arguments.gravity = gravity;
            arguments.radius = radius;

            MPI_Bcast(&arguments, 1, Arguments_TYPE, 0, MPI_COMM_WORLD);

            if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
                space = current_space;
                bodies = current_bodies;
                max_mass = current_max_mass;
                pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            }

            // record calculation time
            auto begin = std::chrono::high_resolution_clock::now();

            // boardcast current information of all bodies
            MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            {
                const ImVec2 p = ImGui::GetCursorScreenPos();

                // do calculations on the target bodies
                pool.update_for_tick(elapse, gravity, space, radius, totalThreadSize, &pool);

                // gather calculation result from all ranks
                // calculate the job size of each rank
                std::vector<int> taskSizeList;
                for(int i=0;i<processSize;i++){
                    int startPosition = (bodies/processSize)*i + std::min(bodies%processSize, i);
                    int endPosition = (bodies/processSize)*(i+1) + std::min(bodies%processSize, i+1)-1;
                    taskSizeList.push_back(endPosition-startPosition+1);
                }

                // calculate the strips of received data packages
                std::vector<int> strips;
                strips.push_back(0);
                for(int i=1;i<processSize;i++){
                    strips.push_back(strips[i-1]+taskSizeList[i-1]);
                }

                // calcualte job size of itself
                int startPosition = (bodies/processSize)*rank + std::min(bodies%processSize, rank);
                int endPosition = (bodies/processSize)*(rank+1) + std::min(bodies%processSize, rank+1)-1;
                int jobSize = endPosition-startPosition+1;

                // rank 0 gather calculated result
                MPI_Gatherv(pool.x.data()+startPosition, jobSize, MPI_DOUBLE, pool.x.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.y.data()+startPosition, jobSize, MPI_DOUBLE, pool.y.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.vx.data()+startPosition, jobSize, MPI_DOUBLE, pool.vx.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.vy.data()+startPosition, jobSize, MPI_DOUBLE, pool.vy.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ax.data()+startPosition, jobSize, MPI_DOUBLE, pool.ax.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ay.data()+startPosition, jobSize, MPI_DOUBLE, pool.ay.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.m.data()+startPosition, jobSize, MPI_DOUBLE, pool.m.data(), taskSizeList.data(), strips.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // record calculation time
                auto end = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                std::cout << pool.size() << " bodies in last " << duration << " nanoseconds" << std::endl;


                // plot calculation results
                for (size_t i = 0; i < pool.size(); ++i) {
                    auto body = pool.get_body(i);
                    auto x = p.x + static_cast<float>(body.get_x());
                    auto y = p.y + static_cast<float>(body.get_y());
                    draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                }

                // printf("\n\n");

            }
            ImGui::End();
        });
    }
    else{
        while(true){

            // receive updated plotting parameters from rank 0
            Arguments arguments;
            MPI_Bcast(&arguments, 1, Arguments_TYPE, 0, MPI_COMM_WORLD);
            current_bodies = arguments.current_bodies;
            current_space = arguments.current_space;
            max_mass = arguments.max_mass;
            elapse = arguments.elapse;
            gravity = arguments.gravity;
            radius = arguments.radius;

            if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
                space = current_space;
                bodies = current_bodies;
                max_mass = current_max_mass;
                pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            }

            // receive current information of all bodies
            MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // do calculations on the target bodies
            pool.update_for_tick(elapse, gravity, space, radius, totalThreadSize, &pool);

            // calcualte job size of itself
            int startPosition = (bodies/processSize)*rank + std::min(bodies%processSize, rank);
            int endPosition = (bodies/processSize)*(rank+1) + std::min(bodies%processSize, rank+1)-1;
            int jobSize = endPosition-startPosition+1;

            // send calculation result
            MPI_Gatherv(pool.x.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.y.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.vx.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.vy.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.ax.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.ay.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.m.data()+startPosition, jobSize, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // if ploting windows closed, rank 0 would then stop all other processes
    if(rank==0){
        std::cout << "aborting all processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Type_free(&Arguments_TYPE);
    MPI_Finalize();

}
