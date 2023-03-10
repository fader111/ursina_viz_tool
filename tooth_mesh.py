from ursina import *
import os, cv2
import sys, numpy as np
from csv_parser_pd_ver2 import set_gen_fr_csv_pd_ver3
from rigid_transform import findRigidTransform
from affine3D_prob import get_rigid
import quaternion
# create a window
app = Ursina(borderless=False)

data = None; i=0


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

def to_local_cs(vec_mass, base_vec=(0,0,0)):
    """ shift massiv of vectors to global center
    """
    return vec_mass - base_vec

def to_global_cs(vec_mass, base_vec=(0,0,0)):
    """ shift entity to the vector back to original position
    """
    return  base_vec - vec_mass

# camera
cam = EditorCamera(rotation=(-90, 0, 0), target_z=-100, position=(-10,0,-25))

# light
L3 = PointLight(parent=cam, x=0, y=0, z=-170, color=color.rgba(80, 80, 80))
L4 = PointLight(parent=cam, x=0, y=0, z=170, color=color.rgba(80, 80, 80))

# light
int_ = 80
# L1 = PointLight(parent=cam, z=-16, color=color.rgba(int_, int_, int_))
# L2 = PointLight(parent=cam, z=16, color=color.rgba(int_, int_, int_))

# cam.z = -3

# Text(text='Mamalana', start_tag='@', end_tag='@', ignore=True)
landmark_cube = Entity(model='cube', color=color.lime, scale=(1, 0.05, 0.05), collider='box', origin_y=-.5)

cyl_base = Entity(model=Cylinder(16, start=-.5), color=color.green, 
        scale=(0.05, 1, 0.05),
        # rotation=(0, 0, 0), 
        # position=(1, -1, 1)
        )


ent_base = Entity(scale=(.1, .1, -.1),  flipped_faces=True, rotation=(0, 0, 0), 
        collider="mesh")

_path21 = "../stl/123957/ToothSurface_21.stl"
_path11 = "../stl/123957/ToothSurface_11.stl"
_path12 = "../stl/123957/ToothSurface_12.stl"


# draw lendmarks
if 0:#for i, lm in enumerate(t1_landmarks[16:]):
    lm/=12
    Entity(model='sphere', color=color.lime, scale=(0.2, 0.2, 0.2),
            position=(lm[0], lm[1], lm[2]),
            # rotation=(0, 0, 0), 
        )
# TODO
# ?????????????? 3 ??????????. ???????????????????? ?????????????????? ?? ????????. ???????? ?? 2D 
# ?????????????? ?????? 3 ?????????? ??????, ?????????? ?????? ?????? ?????????? ??????????. 
# ?????????????????? ?????????????? ?? ?????????????????? ?? ???????????? ????????????. 
# ????????????????????, ????????????????-???? ???? ???????????? ??????????????
point = np.array([      [0,0,0],
                        [1,0,0],
                        [1,1,0]], dtype='float32')
                        
pointR= np.array([      [1,0,0],
                        [1,1,0],
                        [0,1,0]], dtype='float32')

############## ?????????????????? ?????????????? ???????? #########################
t1_landmarks = np.array([   [ -8.45, -24.27,  17.52],
                            [-13.1,  -18.8,   15.63],
                            [-11.27, -20.89,  16.53],
                            [ -6.93, -12.43,  34.59],
                            [-12.47, -20.61,  24.67]], dtype='float32')

t2_landmarks = np.array([   [ -8.83, -22.05,  17.36],
                            [-15.91, -19.8,   17.22],
                            [-13.12, -20.64,  17.44],
                            [ -9.42, -11.54,  35.34],
                            [-12.28, -21.21,  25.61]], dtype='float32')
t1_landmarks = t1_landmarks.reshape(-1, 3)
t2_landmarks = t2_landmarks.reshape(-1, 3)

lm_t1_dw_base = Entity(scale=(1, 1, -1))
lm_t2_dw_base = Entity(scale=(1, 1, -1))
# T1 landmarks
lm_dw_t1 = [Entity(parent=lm_t1_dw_base, model='sphere',
                   # scale=(1, 1, 1), # size of spheres
                   flipped_faces=False,
                   color=color.yellow,
                   position=lm,
                   ) for lm in t1_landmarks]
# T2 landmarks
lm_dw_t2 = [Entity(parent=lm_t2_dw_base, model='sphere',
                   flipped_faces=False,
                   color=color.lime,
                   position=(lm[0], lm[1], lm[2]),
                   ) for lm in t2_landmarks]
# ?????? ??????????????
tooth_12_base = Entity(flipped_faces=True, scale=(1, 1, -1))
# tooth_12_base = Entity()
# tooth_12 = Entity(model=_path12, flipped_faces=True, scale=(1, 1, -1))
tooth_12 = Entity(  parent=tooth_12_base, 
                    model=_path12, 
                    flipped_faces=True )
# cube_base = Entity(scale=(0.2, 5, 0.2))
# cube = Entity(parent=cube_base, model="cube")
# base_point = Entity(scale=(1, 1, -1))
# base_pointR = Entity()
# cube_base.world_rotation = (-24, 52, -40)
# cube_base.world_position = (-2.43608, 2.96955, 0.784798) 

trasform_it = 0
if 1:
    lm1, lm2 = t1_landmarks , t2_landmarks
    # ?????????????? ?????? ???????????? ?????????????????? ?? ?????? ?????????????? ?? ?????????????????? ???? 
    # lm1 = to_local_cs(lm1[:], t1_landmarks[0]) # ?? ?????? ?????????????? ?????????????????? ??????????????????
    # lm2 = to_local_cs(lm2[:], t1_landmarks[0])
    M = get_rigid(lm1, lm2)

    rotM =      M[:,:3] # rotation matrix
    transM =    M[:,3:] # translation matrix
    print(f"\nrotM\n {rotM}")
    print(f"transMM\n {transM}")

    lm_dw_t1_ = [Entity(parent=lm_t1_dw_base, model='sphere',
                    # scale=(1, 1, 1), # size of spheres
                    # flipped_faces=False,
                    color=color.red,
                    position=lm,
                    ) for lm in lm1]
    
    lm_dw_t2_ = [Entity(parent=lm_t2_dw_base, model='sphere',
                # scale=(1, 1, 1), # size of spheres
                # flipped_faces=False,
                color=color.blue,
                position=lm,
                ) for lm in lm2]
    # tooth_12_base.world_position= ( -t1_landmarks[0][0], 
    #                                 -t1_landmarks[0][1],
    #                                 t1_landmarks[0][2])


    from scipy.spatial.transform import Rotation  
    r =  Rotation.from_matrix(rotM)
    theta_z, theta_x, theta_y  = r.as_euler("zxy", degrees=True)
    theta_z = -theta_z
    print(f"eulers  xyz {theta_x, theta_y, theta_z}")



base_point = tooth_12
base_point.visible=1
# base_point = tooth_12
if trasform_it:
    # base_point.rotation = (theta_x, theta_y, theta_z)
    # base_point.rotation = base_point.world_rotation
    base_point.rotation_x = 0
    base_point.rotation_y = 0
    base_point.rotation_z = 0
    base_point.position = (0,0,0)#transM

    # cube_base.world_rotation = (theta_x, theta_y, theta_z)
    # cube_base.world_position = transM
    # ?????????? ???????????????? ?????????????? ?????????? ????????????????:
    # ?? ?????????????? ?? ?????????????????????? ?????? world ?? local ????????????????????
    # cube_base rotation  W Vec3(-24, 52, -40)
    # cube_base pos W Vec3(-2.43608, 2.96955, 0.784798) 
    pass


steps = 200 
angles = np.array([theta_x, theta_y, theta_z])

def update():
    global anima
    # base_point = cube_base
    # cube.rotation_y += time.dt * 100                 # Rotate the cube every time update is called
    # base_point.rotation = base_point.world_rotation
    # tooth_12_base.rotation = (0,0,0)
     
    if not held_keys['shift']:
        base_point.rotation_x += (held_keys[ 'w' ] - held_keys[ 's' ])
        base_point.rotation_y += (held_keys[ 'a' ] - held_keys[ 'd' ])
        base_point.rotation_z += (held_keys[ 'z' ] - held_keys[ 'x' ])
    else:
        base_point.position += (0, (held_keys[ 'w' ] - held_keys[ 's' ]) * time.dt , 0) # move y
        base_point.position += ((held_keys[ 'd' ] - held_keys[ 'a' ]) * time.dt , 0, 0) # move x
        base_point.position += (0,0, (held_keys[ 'z' ] - held_keys[ 'x' ]) * time.dt )  # move z

    if  held_keys['f']: 

        if abs(base_point.rotation_z) <= abs(theta_z):
            base_point.rotation += angles/steps
            base_point.position += transM/steps
        # print(f"rot {base_point.rotation}\n pos {base_point.position}")
        print(f"trans {base_point.transform[:2]}") # position, rotation

    
    if  held_keys['r']:
        base_point.rotation = (0,0,0)
        base_point.position = (0,0,0)

        # print(f"transform {base_point.transform}") 
        # print(f"base_point.rotation_z {base_point.rotation_z}")



    # base.screenshot(
    #             namePrefix = '\\misc\\video_temp\\video_' + str(i).zfill(4) + '.png',
    #             defaultFilename = 0,
    #             )
    

def input(key):
 
    if key == 'escape':
        quit()
 
    # create new entityes interactively    
    if key == 'n':
        Entity(model='cube', color=color.orange, collider='box',
            origin_y=-.5
        )
    if 0:#if key =='mouse right':
        # print(f"cube_base rotation {cube_base.rotation}")
        print(f"cube_base pos {base_point.position}")
        # print(f"cube_base rotation  W {cube_base.world_rotation}")
        print(f"cube_base pos W {base_point.world_position}")
    # if key=='p':
    #     base_point.scale_z+=0.2    
    # if key=='o':
    # base_point.scale_z-=0.2
    if key=="p":    
        print(f"rot {base_point.rotation}\n pos {base_point.position}")
        print(f"cam pos {cam.position}")
    # print(key)
# start running
app.run()

# find buble sorting.