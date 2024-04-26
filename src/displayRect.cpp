#include "displayRect.h"
#include <iostream>
#include <string>
#include <vector>

static const struct
{
    float x, y;
} vertices[6] =
{
    // Bottom-right
    { -1.0f, -1.0f },
    {  1.0f, -1.0f },
    {   1.0f,  1.0f },
    // Top-left
    { -1.0f, -1.0f },
    {  1.0f, 1.0f },
    {   -1.0f,  1.0f },
};
 
static const char* vertex_shader_text =
"#version 110\n"
"attribute vec2 aPos;\n"
"varying vec2 vUV;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(aPos, 0.0, 1.0);\n"
"    vUV = aPos.xy * 0.5 + 0.5;\n"
"}\n";
 
static const char* fragment_shader_text =
"#version 110\n"
"varying vec2 vUV;\n"
"uniform sampler2D uTexture;\n"
"void main()\n"
"{\n"
"    gl_FragColor = vec4(texture2D(uTexture, vUV).rgb, 1.0);\n"
// "    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
"}\n";

static GLuint createShader(GLenum shaderType, const char* sourceCode)
{
    const char* shaderTypeName = shaderType == GL_VERTEX_SHADER ? "vertex" : "fragment";

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &sourceCode, NULL);
    glCompileShader(shader);

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);
        std::string errorLogStr(errorLog.begin(), errorLog.end());

        std::cerr << "Shader error (" << shaderTypeName << "):" << std::endl;
        std::cerr << errorLogStr << std::endl;

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        glDeleteShader(shader); // Don't leak the shader.
        exit(1);
    }

    return shader;
}


DisplayRect::DisplayRect()
{
    // Shader
    GLuint vertex_shader = createShader(GL_VERTEX_SHADER, vertex_shader_text);
    GLuint fragment_shader = createShader(GL_FRAGMENT_SHADER, fragment_shader_text);
 
    _program = glCreateProgram();
    glAttachShader(_program, vertex_shader);
    glAttachShader(_program, fragment_shader);
    glLinkProgram(_program);
 
    glUseProgram(_program);
    _apos_location = glGetAttribLocation(_program, "aPos");
    _tex_location = glGetAttribLocation(_program, "uTexture");

    // Mesh
    glGenBuffers(1, &_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, _vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
 
    glEnableVertexAttribArray(_apos_location);
    glVertexAttribPointer(_apos_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (void*) 0);
}

void DisplayRect::render(GLuint textureId)
{
    glUseProgram(_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glUniform1i(_tex_location, 0);
    // glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}