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
cam = EditorCamera(rotation=(-90, 0, 0), target_z=-170)

# light
L3 = PointLight(parent=cam, x=0, y=0, z=-170, color=color.rgba(80, 80, 80))

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
_path12 = "../stl/123957/ToothSurface_12.stl"

# ent1 = Entity(parent = ent_base, model=_path21)
# ent2 = Entity(parent = ent_base, model=_path11)
# ent1.texture='sky_sunset'
# ent1.collider.visible = True
# print(f"ent1 {ent1._flipped_faces}")

# get landmarks from diego case 123967
# "E:\awsCollectedDataPedro\collected\123957.oas"
# t1_landmarks = set_gen_fr_csv_pd_ver3(r"C:\Projects\torchEncoder\csv\Diego_1k.csv", 
        # one_case="123957.oas")[0][1]
# t1_landmarks = t1_landmarks.reshape(-1,3) # group by 3
# print("len", t1_landmarks)
# draw lendmarks
if 0:#for i, lm in enumerate(t1_landmarks[16:]):
    lm/=12
    Entity(model='sphere', color=color.lime, scale=(0.2, 0.2, 0.2),
            position=(lm[0], lm[1], lm[2]),
            # rotation=(0, 0, 0), 
        )
# TODO
# сделать 3 точки. нарисовать лендмарки с ними. пока в 2D 
# сделать еще 3 точки так, чтобы это был ригид транс. 
# посчитать матрицу и применить к первой тройке. 
# посмотреть, совпадет-ли со второй тройкой
point = np.array([      [0,0,0],
                        [1,0,0],
                        [1,1,0]], dtype='float32')
                        
pointR= np.array([      [1,0,0],
                        [1,1,0],
                        [0,1,0]], dtype='float32')

############## лендмарки вставим сюда #########################
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
# зуб нарисуэ
tooth_12_base = Entity(flipped_faces=True, scale=(1, 1, -1))
# tooth_12 = Entity(model=_path12, flipped_faces=True, scale=(1, 1, -1))
tooth_12 = Entity(parent=tooth_12_base)#, model=_path12 )

# тестовый куб, идея с которым не взлетела
# # cube_base = Entity(scale=(0.2, 5, 0.2))
# # cube = Entity(parent=cube_base, model="cube")
# base_point = Entity(scale=(1, 1, -1))
# base_pointR = Entity()

# ents = [Entity( 
#                 parent=base_point, 
#                 model = 'sphere',
#                 position=i, 
#                 color=color.lime,
#                 scale=(.2, 0.2, 0.2)
#                 ) for i in point]
# entsR = [Entity( 
#                 parent=base_pointR, 
#                 model = 'sphere',
#                 position=i, 
#                 color=color.red,
#                 scale=(.2, 0.2, 0.2)
#                 ) for i in pointR]

# base_point.visible = 1
# base_pointR.visible = 1

# вариант с точками не прошел, преобразование оказалось неуспешно. 
# пробуем с мешами
if 0:    
    en1 = Entity( model='cube', rotation=(0, 0, 0), )
    en2 = Entity( model='cube', rotation=(45, 0, 45), position=(2,1,0))

    en1_pos = en1.position
    en2_pos = en2.position
    print(f"en2_pos {en2_pos}")
trasform_it = 1
if 1:
    n_lm = 3 # landmark number for tests
    assert n_lm<5 , "wrong lm number"
    ret = 1
    M=None
    lm1, lm2 = t1_landmarks , t2_landmarks
    # сначала оба набора лендмарок и зуб двигаем в локальную ск 
    lm1 = to_local_cs(lm1[:], t1_landmarks[0]) # в лок систему координат лендмарки
    lm2 = to_local_cs(lm2[:], t1_landmarks[0])
    if 1:
        ret, M, mask = cv2.estimateAffine3D(lm1, 
                                            lm2, 
                                            confidence = .999999) #это с точками не пашет, хз почему.
    else:
        M = get_rigid(lm1, lm2)
    assert ret ==1, "Wrong rotation matrix got"

    rotM =      M[:,:3] # rotation matrix
    transM =    M[:,3:] # translation matrix
    print(f"\nrotM\n {rotM}")
    print(f"transMM\n {transM}")

    lm_dw_t1_ = [Entity(parent=lm_t1_dw_base, model='sphere',
                    # scale=(1, 1, 1), # size of spheres
                    flipped_faces=False,
                    color=color.red,
                    position=lm,
                    ) for lm in lm1]
    
    lm_dw_t2_ = [Entity(parent=lm_t2_dw_base, model='sphere',
                # scale=(1, 1, 1), # size of spheres
                flipped_faces=False,
                color=color.blue,
                position=lm,
                ) for lm in lm2]
    tooth_12_base.world_position= ( -t1_landmarks[0][0], 
                                    -t1_landmarks[0][1],
                                    t1_landmarks[0][2])
    

    # # rotM, R = qr(rotM)
    # # print(f"Morto= {rotM.dot(rotM.T)}")
    # # сделали кватернион вращения
    # q = quaternion.from_rotation_matrix(rotM)
    # q_vect = quaternion.quaternion(0, t1_landmarks[n_lm][0],  
    #     t1_landmarks[n_lm][1], t1_landmarks[n_lm][2])  
    # qT = q_vect * q.conjugate()
    # qT = np.normalized(qT)
    # angle = 2 * acos(qT.w)
    # x = qT.x / sqrt(1-qT.w*qT.w)
    # y = qT.y / sqrt(1-qT.w*qT.w)
    # z = qT.z / sqrt(1-qT.w*qT.w)

    # rad = 180 / math.pi # thats for 
    # theta_x = x * rad
    # theta_y = y * rad
    # theta_z = z * rad
    
    # print(f"end quat ")

    rad = 180 / math.pi # thats for 
    theta_x = rad * math.atan2(rotM[2][1], rotM[2][2]) 
    theta_y = rad * math.atan2(-rotM[2][0], 
            math.sqrt(rotM[2][1]*rotM[2][1] + rotM[2][2]*rotM[2][2]))
    theta_z = - rad * math.atan2(rotM[1][0], rotM[0][0])



###################### новое мышление ######################
# кватернионы 
# Source: https://stackoverflow.com/a/6802723 .



# v = [3,5,0]
# axis = [4,4,1]
# theta = math.pi #radian

# vector = np.array([0.] + t1_landmarks[2])
# rot_axis = np.array([0.] + t2_landmarks[2])
# axis_angle = (theta*0.5) * rot_axis/np.linalg.norm(rot_axis)
# vec = quaternion.quaternion(*t1_landmarks[2])
# qlog = quaternion.quaternion(*axis_angle)
# q = np.exp(qlog)
# v_prime = q * vec * np.conjugate(q)

# print("v_prime" , v_prime) # quaternion(0.0, 2.7491163, 4.7718093, 1.9162971)

# v_prime_vec = v_prime.imag # [2.74911638 4.77180932 1.91629719] as a numpy array
# print(f" v_prime_vec {v_prime_vec}")

# theta_x, theta_y, theta_z = v_prime_vec


base_point = tooth_12_base
# base_point = tooth_12
if trasform_it:
    base_point.rotation = (theta_x, theta_y, theta_z)
    # base_point.rotation_z = theta_z
    # base_point.rotation_x = theta_x
    # base_point.rotation_y = theta_y
    # base_point.rotation = (0,0,0)
    # base_point.x = transM[0]
    # base_point.y = transM[1]
    # base_point.z = transM[2]
    base_point.position = transM

    cube_base.rotation = (theta_x, theta_y, theta_z)
    cube_base.position = transM
    # после подгонки вручную такое вращение:
    # и вращеие и транясляция для world и local одинаковые
    # cube_base rotation  W Vec3(-24, 52, -40)
    # cube_base pos W Vec3(-2.43608, 2.96955, 0.784798) 

def update():
    global i
    base_point = cube_base
    # cube.rotation_y += time.dt * 100                 # Rotate the cube every time update is called
    # base_point.rotation = base_point.world_rotation
    # tooth_12_base.rotation = (0,0,0)
     
    if held_keys['shift']:
        base_point.rotation_x += (held_keys[ 'w' ] - held_keys[ 's' ])
        base_point.rotation_y += (held_keys[ 'a' ] - held_keys[ 'd' ])
        base_point.rotation_z += (held_keys[ 'z' ] - held_keys[ 'x' ])
    else:
        base_point.position += (0, (held_keys[ 'w' ] - held_keys[ 's' ]) * time.dt , 0) # move y
        base_point.position += ((held_keys[ 'd' ] - held_keys[ 'a' ]) * time.dt , 0, 0) # move x
        base_point.position += (0,0, (held_keys[ 'z' ] - held_keys[ 'x' ]) * time.dt )  # move z
   

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
        print(f"cube_base rotation {cube_base.rotation}")
        print(f"cube_base pos {cube_base.position}")
        print(f"cube_base rotation  W {cube_base.world_rotation}")
        print(f"cube_base pos W {cube_base.world_position}")

# start running
app.run()

# find buble sorting.