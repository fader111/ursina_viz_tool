from ursina import *
import os, cv2
import sys, numpy as np
from csv_parser_pd_ver2 import set_gen_fr_csv_pd_ver3

# create a window
app = Ursina(borderless=False)

data = None; i=0
def update():
    global i
    # cube.rotation_y += time.dt * 100                 # Rotate the cube every time update is called
     
    if held_keys['left shift']:
        landmark_cube.rotation_x += (held_keys[ 'w' ] - held_keys[ 's' ])
        landmark_cube.rotation_y += (held_keys[ 'a' ] - held_keys[ 'd' ])
        landmark_cube.rotation_z += (held_keys[ 'z' ] - held_keys[ 'x' ])
    else:
        landmark_cube.position += (0, (held_keys[ 'w' ] - held_keys[ 's' ]) * time.dt , 0) # move y
        landmark_cube.position += ((held_keys[ 'd' ] - held_keys[ 'a' ]) * time.dt , 0, 0) # move x
        landmark_cube.position += (0,0, (held_keys[ 'z' ] - held_keys[ 'x' ]) * time.dt )  # move z
   

    if held_keys['aw']:                               # If q is pressed
        ent1.position += (0, time.dt, 0)           # move up vertically
        # camera.position.x += time.dt           # move up vertically
    if held_keys['dw']:                               # If a is pressed
        ent1.position -= (0, time.dt, 0) 
    
    if held_keys['zw']:                               # If q is pressed
        ent1.rotation_y += time.dt*10           # move up vertically
        # camera.position.x += time.dt           # move up vertically
    if held_keys['xw']:                               # If a is pressed
        ent1.rotation_y -= time.dt*10

    # base.screenshot(
    #             namePrefix = '\\misc\\video_temp\\video_' + str(i).zfill(4) + '.png',
    #             defaultFilename = 0,
    #             )
    i+=1
    

def input(key):
    if key == '1':
        # ent1.hide() if ent1.visible == True else ent1.show()
        ent_base.visible = False if ent_base.visible == True else True
        # print(f"ent1.visible {ent1.visible}")
    if key == 'escape':
        quit()
    if key == 'v': # video scrinshots
        base.movie(namePrefix=f'\\misc\\video_temp\\video_', duration=2.0, fps=30, format='png', sd=4)
    
    # create new entityes interactively    
    if key == 'n':
        Entity(model='cube', color=color.orange, collider='box',
            origin_y=-.5
        )
    if key =='mouse right':
        print(f"mose click")

def collide():
    def coord_change():
        player.z += (held_keys['w'] - held_keys['s']) * time.dt * 6
        player.x += (held_keys['d'] - held_keys['a']) * time.dt * 6
        player.y += (held_keys['z'] - held_keys['x']) * time.dt * 6

    if player.intersects(ent1).hit:
        player.color = color.lime
        print('player is inside trigger box')

    else:
        player.color = color.gray
        coord_change()  


# debug camera
cam = EditorCamera()

# light
int_ = 80
L1 = PointLight(parent=cam, z=-16, color=color.rgba(int_, int_, int_))
# cam.z = -3

# Text(text='Mamalana', start_tag='@', end_tag='@', ignore=True)
landmark_cube = Entity(model='cube', color=color.lime, scale=(1, 0.05, 0.05), collider='box', origin_y=-.5)

cyl_base = Entity(model=Cylinder(16, start=-.5), color=color.green, 
        scale=(0.05, 1, 0.05),
        rotation=(0, 0, 0), 
        position=(1, -1, 1)
        )


# cyl_base2 = Entity(model=Cylinder(16, start=-.5), color=color.red, 
#         scale=(0.05, 1, 0.05),
#         rotation=(45, 45, 45), 
#         position=(1.1, -1.1, 1)
        # )

for i in range (0):
    # val = Entity(parent=cyl_base, model=Cylinder(16, start=-.5),)
    val = Entity(parent=cyl_base, model=Cylinder(16,))
    # val.radius = 30+i*10
    val.position = (1+i,1+i, 1+i)
    val.rotation = (1+i*10,1+i*10, 1+i)

ent_base = Entity(scale=(.1, .1, -.1),  flipped_faces=True, rotation=(0, 0, 0), 
        collider="mesh")

_path21 = "../stl/123957/ToothSurface_21.stl"
_path11 = "../stl/123957/ToothSurface_11.stl"

ent1 = Entity(parent = ent_base, model=_path21)
ent2 = Entity(parent = ent_base, model=_path11)
ent1.texture='sky_sunset'
# ent1.collider.visible = True
# print(f"ent1 {ent1._flipped_faces}")

# get landmarks from diego case 123967
# "E:\awsCollectedDataPedro\collected\123957.oas"
t1_landmarks = set_gen_fr_csv_pd_ver3(r"C:\Projects\torchEncoder\csv\Diego_1k.csv", 
        one_case="123957.oas")[0][1]
t1_landmarks = t1_landmarks.reshape(-1,3) # group by 3
print("len", t1_landmarks)
# draw lendmarks
for i, lm in enumerate(t1_landmarks[16:]):
    if i>100:
        continue
    lm/=12
    Entity(model='sphere', color=color.lime, scale=(0.2, 0.2, 0.2),
            position=(lm[0], lm[1], lm[2]),
            # rotation=(0, 0, 0), 
        )
# start running
app.run()
