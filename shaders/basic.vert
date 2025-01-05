#version 330 core

layout (location = 0) in vec3 aPos;      // Vertex position
layout (location = 1) in vec3 aColor;    // Vertex color
layout (location = 2) in vec2 aTexCoord; // Texture coords

out vec3 v_Color;
out vec2 v_TexCoord;

uniform mat4 u_MVP;

void main()
{
    gl_Position = u_MVP * vec4(aPos, 1.0);
    v_Color = aColor;
    v_TexCoord = aTexCoord;
}
