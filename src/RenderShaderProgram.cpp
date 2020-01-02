// Copyright 2004-present Facebook. All Rights Reserved.

#include <pangolin/gl/glsl.h>

constexpr const char *shaderText = R"Shader(
@start vertex
#version 330 core

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 3) in vec2 vertexUV;

uniform mat4 MVP;

out vec2 UV;
out vec3 outColor;

void main(){
    gl_Position =  MVP * vec4(vertex,1);
    UV = vertexUV;
    outColor = vertexColor;
}


@start fragment
#version 330 core

in vec2 UV;
in vec3 outColor;

layout (location = 0) out vec4 FragColor;

uniform sampler2D textureSampler;

void main(){
    FragColor = texture2D(textureSampler, UV) * outColor;
}

)Shader";

pangolin::GlSlProgram GetShaderProgram() {
    pangolin::GlSlProgram program;

    program.AddShader(pangolin::GlSlAnnotatedShader, shaderText);
    program.Link();

    return program;
}
