#define STB_IMAGE_IMPLEMENTATION
#include "Utils.h"
#include <CLI/CLI.hpp>
#include <pangolin/geometry/geometry_obj.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>
#include <pangolin/utils/file_utils.h>
#include <random>
#include <stb_image.h>

extern pangolin::GlSlProgram GetShaderProgram();

#define SEED 1337
#define NUM_VIEWS 20

typedef struct {
    GLuint vbId;
    int numTriangles;
    size_t materialId;
} DrawObject;

std::vector<DrawObject> gDrawObjects;

static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
    float v10[3];
    v10[0] = v1[0] - v0[0];
    v10[1] = v1[1] - v0[1];
    v10[2] = v1[2] - v0[2];

    float v20[3];
    v20[0] = v2[0] - v0[0];
    v20[1] = v2[1] - v0[1];
    v20[2] = v2[2] - v0[2];

    N[0] = v20[1] * v10[2] - v20[2] * v10[1];
    N[1] = v20[2] * v10[0] - v20[0] * v10[2];
    N[2] = v20[0] * v10[1] - v20[1] * v10[0];

    float len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
    if (len2 > 0.0f) {
        float len = sqrtf(len2);

        N[0] /= len;
        N[1] /= len;
        N[2] /= len;
    }
}
namespace // Local utility functions
{
struct vec3 {
    float v[3];
    vec3() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    }
};

void normalizeVector(vec3 &v) {
    float len2 = v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2];
    if (len2 > 0.0f) {
        float len = sqrtf(len2);

        v.v[0] /= len;
        v.v[1] /= len;
        v.v[2] /= len;
    }
}

// Check if `mesh_t` contains smoothing group id.
bool hasSmoothingGroup(const tinyobj::shape_t &shape) {
    for (size_t i = 0; i < shape.mesh.smoothing_group_ids.size(); i++) {
        if (shape.mesh.smoothing_group_ids[i] > 0) {
            return true;
        }
    }
    return false;
}

void computeSmoothingNormals(const tinyobj::attrib_t &attrib,
                             const tinyobj::shape_t &shape,
                             std::map<int, vec3> &smoothVertexNormals) {
    smoothVertexNormals.clear();
    std::map<int, vec3>::iterator iter;

    for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
        // Get the three indexes of the face (all faces are triangular)
        tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
        tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
        tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

        // Get the three vertex indexes and coordinates
        int vi[3];     // indexes
        float v[3][3]; // coordinates

        for (int k = 0; k < 3; k++) {
            vi[0] = idx0.vertex_index;
            vi[1] = idx1.vertex_index;
            vi[2] = idx2.vertex_index;
            assert(vi[0] >= 0);
            assert(vi[1] >= 0);
            assert(vi[2] >= 0);

            v[0][k] = attrib.vertices[3 * vi[0] + k];
            v[1][k] = attrib.vertices[3 * vi[1] + k];
            v[2][k] = attrib.vertices[3 * vi[2] + k];
        }

        // Compute the normal of the face
        float normal[3];
        CalcNormal(normal, v[0], v[1], v[2]);

        // Add the normal to the three vertexes
        for (size_t i = 0; i < 3; ++i) {
            iter = smoothVertexNormals.find(vi[i]);
            if (iter != smoothVertexNormals.end()) {
                // add
                iter->second.v[0] += normal[0];
                iter->second.v[1] += normal[1];
                iter->second.v[2] += normal[2];
            } else {
                smoothVertexNormals[vi[i]].v[0] = normal[0];
                smoothVertexNormals[vi[i]].v[1] = normal[1];
                smoothVertexNormals[vi[i]].v[2] = normal[2];
            }
        }

    } // f

    // Normalize the normals, that is, make them unit vectors
    for (iter = smoothVertexNormals.begin(); iter != smoothVertexNormals.end();
         iter++) {
        normalizeVector(iter->second);
    }

} // computeSmoothingNormals
} // namespace

static bool FileExists(const std::string &absFilename) {
    bool ret;
    FILE *fp = fopen(absFilename.c_str(), "rb");
    if (fp) {
        ret = true;
        fclose(fp);
    } else {
        ret = false;
    }

    return ret;
}

static int LoadObjAndConvert(std::vector<DrawObject> *drawObjects,
                             std::vector<tinyobj::material_t> &materials,
                             std::map<std::string, GLuint> &textures,
                             std::string &filename, std::string &baseName) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::string warn;
    std::string err;

    bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                   filename.c_str(), baseName.c_str(), true);
    std::cout << warn << std::endl;
    std::cout << err << std::endl;

    std::cout << shapes.size() << " objects" << std::endl;
    std::cout << materials.size() << " materials" << std::endl;

    float maxDist = BoundingCubeNormalizationFromTinyObj(shapes, attrib, true);

    std::cout << "Max distance: " << maxDist << std::endl;
    if (baseName[baseName.length() - 1] != '/') {
        baseName += "/";
    }

    {
        for (size_t m = 0; m < materials.size(); m++) {
            tinyobj::material_t *mp = &materials[m];
            if (mp->diffuse_texname.length()) {
                if (textures.find(mp->diffuse_texname) == textures.end()) {
                    GLuint textureId;
                    int w, h;
                    int comp;
                    std::string textureFilename = mp->diffuse_texname;

                    if (!FileExists(textureFilename)) {
                        textureFilename = baseName + mp->diffuse_texname;
                        if (!FileExists(textureFilename)) {
                            std::cerr << "Unable to find file: "
                                      << baseName + mp->diffuse_texname
                                      << std::endl;
                            exit(1);
                        }
                    }

                    unsigned char *image = stbi_load(
                        textureFilename.c_str(), &w, &h, &comp, STBI_default);

                    std::cout << "Loaded texture: " << textureFilename
                              << ", w =  " << w << ", h = " << h
                              << ", comp = " << comp << std::endl;
                    glGenTextures(1, &textureId);
                    glBindTexture(GL_TEXTURE_2D, textureId);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                    GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                    GL_LINEAR);

                    if (comp == 3) {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                                     GL_UNSIGNED_BYTE, image);
                    } else if (comp == 4) {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                                     GL_RGBA, GL_UNSIGNED_BYTE, image);
                    } else {
                        assert(0);
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);
                    stbi_image_free(image);
                    textures.insert(
                        std::make_pair(mp->diffuse_texname, textureId));
                }
            }
        }
    }

    materials.push_back(tinyobj::material_t());
    {
        for (size_t s = 0; s < shapes.size(); s++) {
            DrawObject o;
            std::vector<float> buffer;

            std::map<int, vec3> smoothVertexNormals;
            if (hasSmoothingGroup(shapes[s]) > 0) {
                std::cout << "Compute smoothing normal for shape [" << s << "]"
                          << std::endl;
                computeSmoothingNormals(attrib, shapes[s], smoothVertexNormals);
            }

            for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) {
                tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
                tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
                tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

                int currentMaterialId = shapes[s].mesh.material_ids[f];

                if ((currentMaterialId < 0) ||
                    (currentMaterialId >= static_cast<int>(materials.size()))) {
                    currentMaterialId = materials.size() - 1;
                }

                float diffuse[3];
                for (size_t i = 0; i < 3; i++) {
                    diffuse[i] = materials[currentMaterialId].diffuse[i];
                }
                float tc[3][2];

                if (attrib.texcoords.size() > 0) {
                    if ((idx0.texcoord_index < 0) ||
                        (idx1.texcoord_index < 0) ||
                        (idx2.texcoord_index < 0)) {
                        tc[0][0] = 0.0f;
                        tc[0][1] = 0.0f;
                        tc[1][0] = 0.0f;
                        tc[1][1] = 0.0f;
                        tc[2][0] = 0.0f;
                        tc[2][1] = 0.0f;
                    } else {
                        assert(attrib.texcoords.size() >
                               size_t(2 * idx0.texcoord_index + 1));
                        assert(attrib.texcoords.size() >
                               size_t(2 * idx1.texcoord_index + 1));
                        assert(attrib.texcoords.size() >
                               size_t(2 * idx2.texcoord_index + 1));
                        tc[0][0] = attrib.texcoords[2 * idx0.texcoord_index];
                        tc[0][1] =
                            1.0f -
                            attrib.texcoords[2 * idx0.texcoord_index + 1];
                        tc[1][0] = attrib.texcoords[2 * idx1.texcoord_index];
                        tc[1][1] =
                            1.0f -
                            attrib.texcoords[2 * idx1.texcoord_index + 1];
                        tc[2][0] = attrib.texcoords[2 * idx2.texcoord_index];
                        tc[2][1] =
                            1.0f -
                            attrib.texcoords[2 * idx2.texcoord_index + 1];
                    }
                } else {
                    tc[0][0] = 0.0f;
                    tc[0][1] = 0.0f;
                    tc[1][0] = 0.0f;
                    tc[1][1] = 0.0f;
                    tc[2][0] = 0.0f;
                    tc[2][1] = 0.0f;
                }

                float v[3][3];
                for (int k = 0; k < 3; k++) {
                    int f0 = idx0.vertex_index;
                    int f1 = idx1.vertex_index;
                    int f2 = idx2.vertex_index;

                    assert(f0 >= 0);
                    assert(f1 >= 0);
                    assert(f2 >= 0);

                    v[0][k] = attrib.vertices[3 * f0 + k];
                    v[1][k] = attrib.vertices[3 * f1 + k];
                    v[2][k] = attrib.vertices[3 * f2 + k];
                }

                float n[3][3];
                {
                    bool invalidNormalIndex = false;
                    if (attrib.normals.size() > 0) {
                        int nf0 = idx0.normal_index;
                        int nf1 = idx1.normal_index;
                        int nf2 = idx2.normal_index;

                        if ((nf0 < 0) || (nf1 < 0) || (nf2 < 0)) {
                            invalidNormalIndex = true;
                        } else {
                            for (int k = 0; k < 3; k++) {
                                n[0][k] = attrib.normals[3 * nf0 + k];
                                n[1][k] = attrib.normals[3 * nf1 + k];
                                n[2][k] = attrib.normals[3 * nf2 + k];
                            }
                        }
                    } else {
                        invalidNormalIndex = true;
                    }

                    if (invalidNormalIndex && !smoothVertexNormals.empty()) {
                        int f0 = idx0.vertex_index;
                        int f1 = idx1.vertex_index;
                        int f2 = idx2.vertex_index;

                        if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
                            n[0][0] = smoothVertexNormals[f0].v[0];
                            n[0][1] = smoothVertexNormals[f0].v[1];
                            n[0][2] = smoothVertexNormals[f0].v[2];

                            n[1][0] = smoothVertexNormals[f1].v[0];
                            n[1][1] = smoothVertexNormals[f1].v[1];
                            n[1][2] = smoothVertexNormals[f1].v[2];

                            n[2][0] = smoothVertexNormals[f2].v[0];
                            n[2][1] = smoothVertexNormals[f2].v[1];
                            n[2][2] = smoothVertexNormals[f2].v[2];

                            invalidNormalIndex = false;
                        }
                    }

                    if (invalidNormalIndex) {
                        CalcNormal(n[0], v[0], v[1], v[2]);
                        n[1][0] = n[0][0];
                        n[1][1] = n[0][1];
                        n[1][2] = n[0][2];

                        n[2][0] = n[0][0];
                        n[2][1] = n[0][1];
                        n[2][2] = n[0][2];
                    }
                }

                for (int k = 0; k < 3; k++) {
                    buffer.push_back(v[k][0]);
                    buffer.push_back(v[k][1]);
                    buffer.push_back(v[k][2]);

                    buffer.push_back(n[k][0]);
                    buffer.push_back(n[k][1]);
                    buffer.push_back(n[k][2]);

                    float normalFactor = 0.2;
                    float diffuseFactor = 1 - normalFactor;

                    float c[3] = {
                        n[k][0] * normalFactor + diffuse[0] * diffuseFactor,
                        n[k][1] * normalFactor + diffuse[1] * diffuseFactor,
                        n[k][2] * normalFactor + diffuse[2] * diffuseFactor,
                    };

                    float len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                    if (len2 > 0.0f) {
                        float len = sqrtf(len2);
                        c[0] /= len;
                        c[1] /= len;
                        c[2] /= len;
                    }
                    buffer.push_back(c[0] * 0.5 + 0.5);
                    buffer.push_back(c[1] * 0.5 + 0.5);
                    buffer.push_back(c[2] * 0.5 + 0.5);

                    buffer.push_back(tc[k][0]);
                    buffer.push_back(tc[k][1]);
                }
            }

            o.vbId = 0;
            o.numTriangles = 0;

            if (shapes[s].mesh.material_ids.size() > 0 &&
                shapes[s].mesh.material_ids.size() > s) {
                o.materialId = shapes[s].mesh.material_ids[0];
            } else {
                o.materialId = materials.size() - 1;
            }

            if (buffer.size() > 0) {
                glGenBuffers(1, &o.vbId);
                glBindBuffer(GL_ARRAY_BUFFER, o.vbId);
                glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float),
                             &buffer.at(0), GL_STATIC_DRAW);

                // 3:vtx, 3:normal, 3:col, 2:texcoord
                o.numTriangles = buffer.size() / (3 + 3 + 3 + 2) / 3;
            }

            drawObjects->push_back(o);
        }
    }

    return maxDist;
}

static void Draw(const std::vector<DrawObject> &drawObjects,
                 std::vector<tinyobj::material_t> &materials,
                 std::map<std::string, GLuint> &textures) {
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_FILL);
    // glEnable(GL_POLYGON_OFFSET_FILL);
    // glPolygonOffset(1.0, 1.0);

    GLsizei stride = (3 + 3 + 3 + 2) * sizeof(float);

    for (size_t i = 0; i < drawObjects.size(); i++) {
        DrawObject o = drawObjects[i];
        if (o.vbId < 1) {
            continue;
        }
        glBindBuffer(GL_ARRAY_BUFFER, o.vbId);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glBindTexture(GL_TEXTURE_2D, 0);
        if (o.materialId < materials.size()) {
            std::string diffuseTexname =
                materials[o.materialId].diffuse_texname;
            if (textures.find(diffuseTexname) != textures.end()) {
                glBindTexture(GL_TEXTURE_2D, textures[diffuseTexname]);
            }
        }

        glVertexPointer(3, GL_FLOAT, stride, (const void *)0);
        glNormalPointer(GL_FLOAT, stride, (const void *)(sizeof(float) * 3));
        glColorPointer(3, GL_FLOAT, stride, (const void *)(sizeof(float) * 6));
        glTexCoordPointer(2, GL_FLOAT, stride,
                          (const void *)(sizeof(float) * 9));
        glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

int main(int argc, char **argv) {
    std::string directory;
    std::string fileName;
    std::string outputPath;

    CLI::App app{"RenderImages"};

    app.add_option("-d", directory, "Directory with the input data to render.")
        ->required();
    app.add_option("-f", fileName, "File with the input data to render.")
        ->required();
    app.add_option("-o", outputPath, "Output folder with the output data.")
        ->required();

    CLI11_PARSE(app, argc, argv);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

    size_t w = 640;
    size_t h = 480;

    // std::mt19937 generator(SEED);
    pangolin::CreateWindowAndBind("Main", 640, 480);

    // pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);
    pangolin::GlSlProgram prog = GetShaderProgram();

    pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
    pangolin::GlTexture normals(w, h, GL_RGBA32F);
    pangolin::GlTexture vertices(w, h, GL_RGBA32F);
    pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

    std::vector<tinyobj::material_t> materials;
    std::map<std::string, GLuint> textures;

    float maxDist = LoadObjAndConvert(&gDrawObjects, materials, textures,
                                      fileName, directory);

    std::vector<Eigen::Vector3f> views =
        EquiDistPointsOnSphere(NUM_VIEWS, maxDist * 1.1);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w, h, 420, 420, 320, 240, 0.05, 1000),
        pangolin::ModelViewLookAt(0.5, 1.5, 1.5, 0, 0, 0, pangolin::AxisY));
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, 640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);

        d_cam.Activate(s_cam);
        prog.Bind();
        prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
        prog.SetUniform("V", s_cam.GetModelViewMatrix());
        Draw(gDrawObjects, materials, textures);
        prog.Unbind();

        pangolin::FinishFrame();
    }

    // // for (unsigned int v = 0; v < views.size(); v++) {
    // //     s_cam.SetModelViewMatrix(
    // //         pangolin::ModelViewLookAt(
    // //             views[v][0],
    // //             views[v][1],
    // //             views[v][2],
    // //             0,
    // //             0,
    // //             0,
    // //             pangolin::AxisY
    // //         )
    // //     );
    // //     glViewport(0, 0, w, h);
    // //     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // //     d_cam.Activate(s_cam);

    // //     framebuffer.Bind();
    // //     prog.Bind();
    // //     prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
    // //     prog.SetUniform("V", s_cam.GetModelViewMatrix());
    // //     // prog.SetUniform("ToWorld",
    // s_cam.GetModelViewMatrix().Inverse());
    // //     // prog.SetUniform("slant_thr", -1.0f, 1.0f);
    // //     // prog.SetUniform("ttt", 1.0, 0.0, 0, 1);
    // //     pangolin::GlDraw(prog, gl_geom, nullptr);
    // //     pangolin::FinishFrame();
    // //     prog.Unbind();
    // //     framebuffer.Unbind();
    // //     vertices.Save(outputPath, false);

    // // }

    pangolin::QuitAll();

    return 0;
}