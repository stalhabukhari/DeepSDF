#include <CLI/CLI.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/geometry/geometry_obj.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <random>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

#define SEED 1337
#define NUM_VIEWS 20


int main(int argc, char** argv) {
    std::string inputMeshFile;
    std::string outputPath;


    CLI::App app{"RenderImages"};

    app.add_option(
        "-i", 
        inputMeshFile, 
        "File with the input data to render."
    )->required();
    app.add_option(
        "-o", 
        outputPath, 
        "Output folder with the output data."
    )->required();

    CLI11_PARSE(app, argc, argv);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

    pangolin::Geometry geom = pangolin::LoadGeometry(inputMeshFile);
    std::cout << geom.objects.size() << " objects" << std::endl;
    std::cout << geom.textures.size() << " textures" << std::endl;

    {
        int totalNumFaces = 0;
        for(const auto& object: geom.objects) {
            auto itVertIndices = object.second.attributes.find("vertex_indices");
            if (itVertIndices != object.second.attributes.end()) {
                pangolin::Image<uint32_t> ibo = 
                    pangolin::get<pangolin::Image<uint32_t>>(
                        itVertIndices->second
                    );
                totalNumFaces += ibo.h;
            }
        }

        pangolin::ManagedImage<uint8_t> newBuffer(
            3 * sizeof(uint32_t), totalNumFaces
        );
        pangolin::Image<uint32_t> newIbo = 
            newBuffer.UnsafeReinterpret<uint32_t>().SubImage(
                0, 0, 3, totalNumFaces
            );

        int index = 0;
        for(const auto& object: geom.objects) {
            auto itVertIndices = object.second.attributes.find("vertex_indices");
            if (itVertIndices != object.second.attributes.end()) {
                pangolin::Image<uint32_t> ibo = 
                    pangolin::get<pangolin::Image<uint32_t>>(
                        itVertIndices->second
                    );

                for (int i = 0; i < ibo.h; ++i) {
                    newIbo.Row(index).CopyFrom(ibo.Row(i));
                    ++index;
                }
            }
        }

        geom.objects.clear();
        auto faces = geom.objects.emplace(
            std::string("mesh"), pangolin::Geometry::Element()
        );
        faces->second.Reinitialise(3 * sizeof(uint32_t), totalNumFaces);
        faces->second.CopyFrom(newBuffer);

        newIbo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(
            0, 0, 3, totalNumFaces
        );
        faces->second.attributes["vertex_indices"] = newIbo;
    }

    // geom.textures.clear();

    pangolin::Image<uint32_t> modelFaces = 
        pangolin::get<pangolin::Image<uint32_t>>(
            geom.objects.begin()->second.attributes["vertex_indices"]
        );

    float maxDist = BoundingCubeNormalization(geom, true);

    size_t w = 640;
    size_t h = 480;

    std::mt19937 generator(SEED);
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    
    pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);
    pangolin::GlSlProgram prog = GetShaderProgram();

    pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
    pangolin::GlTexture normals(w, h, GL_RGBA32F);
    pangolin::GlTexture vertices(w, h, GL_RGBA32F);
    pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

    std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(
        NUM_VIEWS, maxDist * 1.1
    );

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w, h, 420, 420, 320, 240, 0.05, 1000),
        pangolin::ModelViewLookAt(0.5, 1.5, 1.5, 0, 0, 0, pangolin::AxisY)
    );
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, 640.0f / 480.0f)
        .SetHandler(&handler);

    while(!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        prog.Bind();
        prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
        prog.SetUniform("V", s_cam.GetModelViewMatrix());

        pangolin::GlDraw(prog, gl_geom, nullptr);
        prog.Unbind();

        pangolin::FinishFrame();
    }

    // for (unsigned int v = 0; v < views.size(); v++) {
    //     s_cam.SetModelViewMatrix(
    //         pangolin::ModelViewLookAt(
    //             views[v][0], 
    //             views[v][1], 
    //             views[v][2], 
    //             0, 
    //             0, 
    //             0, 
    //             pangolin::AxisY
    //         )
    //     );
    //     glViewport(0, 0, w, h);
    //     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //     d_cam.Activate(s_cam);

    //     framebuffer.Bind();
    //     prog.Bind();
    //     prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
    //     prog.SetUniform("V", s_cam.GetModelViewMatrix());
    //     // prog.SetUniform("ToWorld", s_cam.GetModelViewMatrix().Inverse());
    //     // prog.SetUniform("slant_thr", -1.0f, 1.0f);
    //     // prog.SetUniform("ttt", 1.0, 0.0, 0, 1);
    //     pangolin::GlDraw(prog, gl_geom, nullptr);
    //     pangolin::FinishFrame();
    //     prog.Unbind();
    //     framebuffer.Unbind();
    //     vertices.Save(outputPath, false);

    // }


    pangolin::QuitAll();

    return 0;

}