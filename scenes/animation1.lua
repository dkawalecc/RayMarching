
-- MIRROR SPHERE
material_props(1, 0) -- mirror
shade_albedo(1, 1, 1); -- white
sphere = primitive_sphere(0, 0, 0, 1)

phase = 0

function update(dt)
    set_position(sphere, 0, math.sin(phase), 0)
    phase = phase + dt * 8
end