#version 320 es
precision lowp float;

layout (location = 0) in vec3 vertexPosition;

uniform mat4 mvpMatrix;   // Model-View-Projection matrix
uniform mat4 mvMatrix;    // Model-View matrix
uniform vec3 u_ColorNear; // Color when close to the camera
uniform vec3 u_ColorFar;  // Color when far from the camera
uniform float u_MaxDistance; // Distance at which the color is fully u_ColorFar

out vec3 Color;

void main()
{
    gl_PointSize = 1.0;
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);

    // Transform the vertex position into view space
    vec4 viewPosition = mvMatrix * vec4(vertexPosition, 1.0);

    // Compute the distance from the camera (origin) to the vertex in view space
    float dist = length(viewPosition.xyz);

    // Normalize the distance to a range between 0 and 1
    float t = clamp(dist / u_MaxDistance, 0.0, 1.0);

    // Interpolate between the near and far colors
    Color = mix(u_ColorFar,u_ColorNear, t);
}