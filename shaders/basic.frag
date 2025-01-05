#version 330 core

in vec3 v_Color;
in vec2 v_TexCoord;

out vec4 FragColor;

// Normal rendering uniforms
uniform vec4 u_Color;
uniform sampler2D u_Texture;

// Picking mode uniform
uniform bool u_PickingMode;
uniform vec3 u_PickingColor;

void main()
{
    if (u_PickingMode) {
        // Output only the picking color (RGB). 
        // Use alpha=1.0 for convenience.
        FragColor = vec4(u_PickingColor, 1.0);
    }
    else {
        // Normal rendering path
        vec4 texColor = texture(u_Texture, v_TexCoord);
        FragColor = vec4(v_Color, 1.0) * u_Color * texColor;
    }
}
