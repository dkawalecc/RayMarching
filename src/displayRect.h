#ifndef DISPLAY_RECT_H
#define DISPLAY_RECT_H

#include "openGL.h"

class DisplayRect
{
    private:
        GLuint _vertex_buffer, _program;
        GLint _tex_location, _apos_location;

    public:
        DisplayRect();

        void render(GLuint textureId);
};

#endif //DISPLAY_RECT_H