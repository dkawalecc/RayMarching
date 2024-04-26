-- FLOOR
material_props(0.8, 0)
shade_albedo(0.8, 0.8, 0.8) -- gray
-- shade_albedo(1, 1, 1); -- white
primitive_plane(0, -25, 0, 0, 1, 0)

-- MIRROR SPHERE
material_props(0.1, 0)
shade_albedo(1, 1, 1) -- white
primitive_sphere(0, 1, 0, 2)

-- Spheres
-- center 0, 0, 0
material_props(0.5, 0) 
AMOUNT_OF_SPHERES = 6
RADIUS = 1
SCENE_RADIUS = 4
spheres = {}
angle=0

for i = 0, AMOUNT_OF_SPHERES-1, 1
do
    shade_albedo(0.6, i/AMOUNT_OF_SPHERES*1.0, 0.6)
    angle = i * 2 * math.pi /AMOUNT_OF_SPHERES
    spheres[i] = primitive_sphere(SCENE_RADIUS*math.cos(angle), 0.2, SCENE_RADIUS*math.sin(angle), RADIUS) 
end

phase = 0
function update(dt)
    for i =  0, AMOUNT_OF_SPHERES-1, 1
    do  
        angle = i * 2 * math.pi /AMOUNT_OF_SPHERES
        set_position(spheres[i], SCENE_RADIUS*math.cos(angle), 2 + math.sin(phase + (i*2*math.pi/AMOUNT_OF_SPHERES)), SCENE_RADIUS*math.sin(angle))
    end
    phase = phase + dt * 4
end