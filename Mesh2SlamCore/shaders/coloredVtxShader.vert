#version 320 es
precision lowp float;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexColor;

uniform mat4 mvpMatrix;
out vec3 Color;
void main()
{
	gl_PointSize = 1.0;
	gl_Position = mvpMatrix*vec4(vertexPosition,1.0);
	Color = vertexColor;
}