#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

static const struct
{
    float x, y;
} vertices[6] =
{
    // Bottom-right
    { -0.6f, -0.6f },
    {  0.6f, -0.6f },
    {  0.0f,  0.7f },
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
"void main()\n"
"{\n"
"    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
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


static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}
 
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main(int argc, char* argv[])
{
    GLFWwindow* window;
 
    glfwSetErrorCallback(error_callback);
 
    if (!glfwInit())
        exit(EXIT_FAILURE);
 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
    window = glfwCreateWindow(1024, 1024, "Simple Triangle", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
 
    glfwSetKeyCallback(window, key_callback);
 
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);
 
    // NOTE: OpenGL error checks have been omitted for brevity

    // Shader
    GLuint vertex_shader = createShader(GL_VERTEX_SHADER, vertex_shader_text);
    GLuint fragment_shader = createShader(GL_FRAGMENT_SHADER, fragment_shader_text);
 
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
 
    glUseProgram(program);
 
    // Mesh
    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
 
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (void*) 0);

    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        // mat4x4 m, p, mvp;
 
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
 
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
 
        // mat4x4_identity(m);
        // mat4x4_rotate_Z(m, m, (float) glfwGetTime());
        // mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        // mat4x4_mul(mvp, p, m);
    
        glUseProgram(program);
        glDrawArrays(GL_TRIANGLES, 0, 3);
 
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
 
    glfwDestroyWindow(window);
 
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
 