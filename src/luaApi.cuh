#ifndef LUA_API_H
#define LUA_API_H

#include <iostream>
#include <stdio.h>
#include <string>
#include <variant>
#include <memory>
#include <sstream>

extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include "scene.cuh"

class LuaRuntime;

extern std::unique_ptr<LuaRuntime> luaRuntime;

int lua_wobble(lua_State* L);
int lua_material_props(lua_State* L);
int lua_primitive_sphere(lua_State* L);
int lua_primitive_plane(lua_State* L);
int lua_primitive_box(lua_State* L);
int lua_shade_albedo(lua_State* L);
int lua_shade_emissive(lua_State* L);
int lua_set_position(lua_State* L);

struct RuntimeContext {
    float3 latestColor;
    bool latestEmissive;
    float latestRoughness;
    float latestMetallic;
};

class LuaRuntime
{
    private:
        std::string _sceneFilepath;
        lua_State *L = nullptr;

    public:
        Scene _scene;
        Scene* d_scene;
        RuntimeContext _ctx;

        LuaRuntime(const char* name)
        {
            std::stringstream ss("");

            ss << "./scenes/";
            ss << name;
            ss << ".lua";

            _sceneFilepath = ss.str();

            cudaMalloc(&d_scene, sizeof(Scene));
        }

        ~LuaRuntime()
        {
            if (L != nullptr) {
                lua_close(L);
            }

            cudaFree(d_scene);
        }

        void runScene()
        {
            _scene = {};
            std::cout << "Executing " << _sceneFilepath << std::endl;

            // Closing previous instance
            if (L != nullptr) {
                lua_close(L);
            }

            // opens Lua
            L = luaL_newstate(); 
            // opens the standard libraries
            luaL_openlibs(L);

            // register functions
            lua_register(L, "wobble", lua_wobble);
            lua_register(L, "material_props", lua_material_props);
            lua_register(L, "shade_albedo", lua_shade_albedo);
            lua_register(L, "shade_emissive", lua_shade_emissive);

            lua_register(L, "primitive_sphere", lua_primitive_sphere);
            lua_register(L, "primitive_plane", lua_primitive_plane);
            lua_register(L, "primitive_box", lua_primitive_box);

            lua_register(L, "set_position", lua_set_position);
            
            // load and execute a Lua script
            if (luaL_dofile(L, _sceneFilepath.c_str()) != LUA_OK) {
                const char* errorMsg = lua_tostring(L, -1);
                std::cout << "Lua Error: " << errorMsg << std::endl;
                lua_pop(L, 1);
            }

            cudaMemcpy(d_scene, &_scene, sizeof(Scene), cudaMemcpyHostToDevice);
        }

        /**
         * Call a function `update' defined in Lua, if it exists
         */
        void callUpdate(double deltaTime)
        {
            /* push functions and arguments */
            lua_getglobal(L, "update");  /* function to be called */
            if (!lua_isfunction(L, -1))
            {
                lua_pop(L, 1);
                return;
            }

            lua_pushnumber(L, deltaTime);   /* push 1st argument */
            
            /* do the call (1 arguments, no result) */
            if (lua_pcall(L, 1, 0, 0) != 0)
                std::cerr << "error running function `update`: " <<
                        lua_tostring(L, -1) << std::endl;
        }

        void setTime(double time)
        {
            _scene.time = time;
            cudaMemcpy(&d_scene->time, &_scene.time, sizeof(double), cudaMemcpyHostToDevice);
        }

        
        void setPrimitivePosition(uint8_t key, float3 position)
        {
            _scene.primitives[key].position = position;
            _scene.primitives[key].bounds.aabb.pos = position;

            cudaMemcpy(&d_scene->primitives[key].position, &position, sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_scene->primitives[key].bounds.aabb.pos, &position, sizeof(float3), cudaMemcpyHostToDevice);
        }
};

int lua_wobble(lua_State* L) {
    Scene& scene = luaRuntime->_scene;

    scene.wobbleAmplitude = lua_tonumber(L, 1);
    scene.wobbleFrequency = lua_tonumber(L, 2);
    scene.wobbleScroll = lua_tonumber(L, 3);

    return 0;
}

int lua_material_props(lua_State* L) {
    RuntimeContext& ctx = luaRuntime->_ctx;

    ctx.latestRoughness = lua_tonumber(L, 1);
    ctx.latestMetallic = lua_tonumber(L, 2);

    return 0;
}

int lua_shade_albedo(lua_State* L) {
    RuntimeContext& ctx = luaRuntime->_ctx;

    ctx.latestColor.x = lua_tonumber(L, 1);
    ctx.latestColor.y = lua_tonumber(L, 2);
    ctx.latestColor.z = lua_tonumber(L, 3);
    ctx.latestEmissive = false;

    return 0;
}

int lua_shade_emissive(lua_State* L) {
    RuntimeContext& ctx = luaRuntime->_ctx;

    ctx.latestColor.x = lua_tonumber(L, 1);
    ctx.latestColor.y = lua_tonumber(L, 2);
    ctx.latestColor.z = lua_tonumber(L, 3);
    ctx.latestEmissive = true;

    return 0;
}


int lua_primitive_sphere(lua_State* L) {
    Scene& scene = luaRuntime->_scene;
    uint32_t key = scene.primitivesCount;
    RuntimeContext& ctx = luaRuntime->_ctx;
    Primitive& primitive = scene.primitives[scene.primitivesCount++];

    double radius = lua_tonumber(L, 4);

    primitive.type = PrimitiveType::SPHERE;
    primitive.position.x = lua_tonumber(L, 1);
    primitive.position.y = lua_tonumber(L, 2);
    primitive.position.z = lua_tonumber(L, 3);
    primitive.data.sphere.radius = radius;

    primitive.material.color = ctx.latestColor;
    primitive.material.emissive = ctx.latestEmissive;
    primitive.material.roughness = ctx.latestRoughness;
    primitive.material.metalic = ctx.latestMetallic;

    primitive.bounds.aabb.pos = primitive.position;
    primitive.bounds.aabb.rad.x = radius;
    primitive.bounds.aabb.rad.y = radius;
    primitive.bounds.aabb.rad.z = radius;

    std::cout << "Created sphere primitive #" << key << std::endl;

    lua_pushnumber(L, key);  /* push key */
    return 1; // 1 result
}


int lua_primitive_plane(lua_State* L) {
    Scene& scene = luaRuntime->_scene;
    uint32_t key = scene.primitivesCount;
    RuntimeContext& ctx = luaRuntime->_ctx;
    Primitive& primitive = scene.primitives[scene.primitivesCount++];

    float3 normal = {
        lua_tonumber(L, 4),
        lua_tonumber(L, 5),
        lua_tonumber(L, 6)
    };

    primitive.type = PrimitiveType::PLANE;
    primitive.position.x = lua_tonumber(L, 1);
    primitive.position.y = lua_tonumber(L, 2);
    primitive.position.z = lua_tonumber(L, 3);
    primitive.data.plane.normal = normal;

    primitive.material.color = ctx.latestColor;
    primitive.material.emissive = ctx.latestEmissive;
    primitive.material.roughness = ctx.latestRoughness;
    primitive.material.metalic = ctx.latestMetallic;

    primitive.bounds.plane.pos = primitive.position;
    primitive.bounds.plane.normal = normal;

    std::cout << "Created plane primitive #" << key << std::endl;

    lua_pushnumber(L, key);  /* push key */
    return 1; // 1 result
}


int lua_primitive_box(lua_State* L) {
    Scene& scene = luaRuntime->_scene;
    uint32_t key = scene.primitivesCount;
    RuntimeContext& ctx = luaRuntime->_ctx;
    Primitive& primitive = scene.primitives[scene.primitivesCount++];

    primitive.type = PrimitiveType::BOX;
    primitive.position.x = lua_tonumber(L, 1);
    primitive.position.y = lua_tonumber(L, 2);
    primitive.position.z = lua_tonumber(L, 3);
    primitive.data.box.size.x = lua_tonumber(L, 4);
    primitive.data.box.size.y = lua_tonumber(L, 5);
    primitive.data.box.size.z = lua_tonumber(L, 6);

    primitive.material.color = ctx.latestColor;
    primitive.material.emissive = ctx.latestEmissive;
    primitive.material.roughness = ctx.latestRoughness;
    primitive.material.metalic = ctx.latestMetallic;

    primitive.bounds.aabb.pos = primitive.position;
    primitive.bounds.aabb.rad = primitive.data.box.size;

    std::cout << "Created box primitive #" << key << std::endl;

    lua_pushnumber(L, key);  /* push key */
    return 1; // 1 result
}

int lua_set_position(lua_State* L) {
    uint8_t key = lua_tonumber(L, 1);

    luaRuntime->setPrimitivePosition(key, {
        (float) lua_tonumber(L, 2),
        (float) lua_tonumber(L, 3),
        (float) lua_tonumber(L, 4)
    });

    std::cout << "Moved #" << key << std::endl;

    return 0; // no result
}

#endif //LUA_API_H