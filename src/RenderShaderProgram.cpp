// Copyright 2004-present Facebook. All Rights Reserved.

#include <pangolin/gl/glsl.h>

constexpr const char* shaderText = R"Shader(
@start vertex
#version 330 core

layout(location = 0) in vec3 vertex;

uniform mat4 MVP;

attribute vec2 uv;
varying vec2 vUV;

void main(){

    gl_Position =  MVP * vec4(vertex,1);
    vUV = uv;

}


@start fragment
#version 330 core

varying vec2 vUV;
uniform sampler2D texture_76;

void main(){
    gl_FragColor = texture2D(texture_76, vUV);
}
)Shader";

pangolin::GlSlProgram GetShaderProgram() {
  pangolin::GlSlProgram program;

  program.AddShader(pangolin::GlSlAnnotatedShader, shaderText);
  program.Link();

  return program;
}
