""" Tool for visualize the ML model inference, staging process animation
"""
import os
import sys
from ursina import *
from csv_parser_pd_ver2 import set_gen_fr_csv_pd_ver3
import cv2
import numpy as np
from affine3D_prob import get_rigid
from scipy.spatial.transform import Rotation


up_teeth_nums16 = [18, 17, 16, 15, 14, 13, 12, 11,
                   21, 22, 23, 24, 25, 26, 27, 28]
dw_teeth_nums16 = [38, 37, 36, 35, 34, 33, 32, 31,
                   41, 42, 43, 44, 45, 46, 47, 48]

case_folder = "123957"
mesh_paths_up = [
    f"../stl/{case_folder}/ToothSurface_{n}.stl" for n in up_teeth_nums16]
mesh_paths_dw = [
    f"../stl/{case_folder}/ToothSurface_{n}.stl" for n in dw_teeth_nums16]

asset_folder = os.path.dirname(sys.argv[0])
case_folder_fpath = os.path.join(asset_folder, f"../stl/{case_folder}")
stl_flist = os.listdir(case_folder_fpath)  # ['*.stl', .... ]


def rm_miss_fnames(names_lst):
    ''' remove file names from list if there is no tooth file
    '''
    for pth in names_lst[:]:
        if not pth.split("/")[-1] in stl_flist:
            names_lst.remove(pth)


for path in mesh_paths_dw, mesh_paths_up:
    rm_miss_fnames(path)

# data
case = "123957.oas"
csv_path = r"C:\Projects\torchEncoder\csv\diego_landmarks_over_900.csv"
ml_vec_points = 480  # x,y,z X 5 landmarks X 16 teeth X 2 jaws  - length of vector from NN
impl_trans = True  # go transform

# create a window
window.title = "Ortho viewer V0.1 - " + case  # с какого это не работает?
app = Ursina(development_mode=True, borderless=False)

# camera
cam = EditorCamera(rotation=(-90, 0, 0), target_z=-170)

# light
L1 = PointLight(parent=cam, x=0, y=0, z=-170, color=color.rgba(80, 80, 80))

##################################### draw teeth #####################################
tooth_ind = 0


def action():
    global tooth_ind
    try:
        obj_ = mouse.hovered_entity
        tooth_ind = ind = jaw_up.index(obj_)
        mouse.hovered_entity.texture = 'sky_sunset'
        # ToothSurface_11.stl
        print(f"click on {str(mouse.hovered_entity.model).split('/')[-1]}")
        print(f"index =  {ind}")
    except:
        pass


# Entities
# parent entity to control the group of entities
jaw_up_base = Entity(flipped_faces=True, scale=(1, 1, -1))
jaw_dw_base = Entity(flipped_faces=True, scale=(1, 1, -1))

# child (real) entityes
# on_click doesn't work without collider
jaw_up = [Entity(parent=jaw_up_base, model=tooth, on_click=action,
                 collider="box") for tooth in mesh_paths_up]
jaw_dw = [Entity(parent=jaw_dw_base, model=tooth, on_click=action,
                 collider="box") for tooth in mesh_paths_dw]

##################################### draw landmarks ##################################

# obtain landmarks
landmarks = set_gen_fr_csv_pd_ver3(csv_path, one_case=case)[0]
t1_landmarks = landmarks[0]
t2_landmarks = landmarks[1]

# reshape landmarks by x,y,z coordinates
# group them by 3
t1_landmarks = t1_landmarks.reshape(-1, 3)
t2_landmarks = t2_landmarks.reshape(-1, 3)
# t2_landmarks = t2_landmarks[6*5:7*5]
# print("len", t1_landmarks)

# draw lower jaw first
# base landmark need for control all batch
lm_t1_dw_base = Entity(scale=(1, 1, -1))
lm_t2_dw_base = Entity(scale=(1, 1, -1))
# T1 landmarks
lm_dw_t1 = [Entity(parent=lm_t1_dw_base, model='sphere',
                   # scale=(1, 1, 1), # size of spheres
                   flipped_faces=False,
                   color=color.yellow,
                   position=lm,
                   ) for lm in t1_landmarks[:ml_vec_points//2//3]]
# T2 landmarks
lm_dw_t2 = [Entity(parent=lm_t2_dw_base, model='sphere',
                   flipped_faces=False,
                   color=color.lime,
                   position=(lm[0], lm[1], lm[2]),
                   ) for lm in t2_landmarks[:ml_vec_points//2//3]]

# Upper jaw
lm_t1_up_base = Entity(scale=(1, 1, -1))
lm_t2_up_base = Entity(scale=(1, 1, -1))
lm_up_t1 = [Entity(parent=lm_t1_up_base, model='sphere',
                   flipped_faces=False,
                   color=color.red,
                   # scale=(.1, .1, .1),
                   position=(lm[0], lm[1], lm[2]),
                   ) for lm in t1_landmarks[ml_vec_points//2//3:]]

lm_up_t2 = [Entity(parent=lm_t2_up_base, model='sphere',
                   flipped_faces=False,
                   color=color.green,
                   # scale=(.1, .1, .1),
                   position=(lm[0], lm[1], lm[2]),
                   ) for lm in t2_landmarks[ml_vec_points//2//3:]]


##################################### transform jaws ##################################

# reshape landmarks for iteration by teeth
# divide by teeth - each tooth has 5 landmarks
t1_landmarks = t1_landmarks.reshape(-1, 5, 3)
t2_landmarks = t2_landmarks.reshape(-1, 5, 3)

# called in input


def transform(do_tr=True):
    ''' rigid transform upper and lower jaw based on landmarks '''

    # upd_lm_visibility()

    # for jaw in jaw_up[:-8],: #№ короткая версия
    for jaw in jaw_dw, jaw_up:
        t1_lm_jaw = t1_landmarks[:16] if jaw == jaw_dw else t1_landmarks[16:]
        t2_lm_jaw = t2_landmarks[:16] if jaw == jaw_dw else t2_landmarks[16:]
        for t1_lm_tooth, t2_lm_tooth, tooth in zip(t1_lm_jaw,
                                                   t2_lm_jaw,
                                                   jaw
                                                   ):  # 16 teeth
            # rigid transform 
            M = get_rigid(t1_lm_tooth, t2_lm_tooth)

            rotM = M[:, :3]  # rotation matrix
            transM = M[:, 3:]  # translation matrix

            print(f"rotM \n{rotM}")
            print(f"transM \n{transM}")
            print(f"det: {np.linalg.det(rotM)}")
            print(f"t1_lm_tooth t2_lm_tooth \n{t1_lm_tooth} \n{t2_lm_tooth}")

            # calc Euler angles from rotation matrix:
            #     | r11 | r12 | r13 | theta x = atan2(r32, r33)
            # M = | r21 | r22 | r23 | theta y = atan2(-r31, sqrt(r32^2 + r33^2))
            #     | r31 | r32 | r33 | theta z = atan2(r21, r11)
            # rad = 180 / math.pi  # thats for
            # это не работает. работает вариант скайпи см нижеa
            # theta_x = rad * math.atan2(M[2][1], M[2][2])
            # theta_y = rad * math.atan2(-M[2][0],
            #                            math.sqrt(M[2][1]*M[2][1] + M[2][2]*M[2][2]))
            # # NOTE minus z rotation is Ursina feature, not a bug
            # theta_z = - rad * math.atan2(M[1][0], M[0][0])

            # that works
            r = Rotation.from_matrix(rotM)
            theta_z, theta_x, theta_y = r.as_euler("zxy", degrees=True)
            theta_z = -theta_z  # ursina property

            # implement calculated values rotation and movement to teeth
            if do_tr:
                tooth.rotation = (theta_x, theta_y, theta_z)
                tooth.position = transM
            else:
                tooth.rotation = (0, 0, 0)
                tooth.position = (0, 0, 0)


def upd_lm_visibility():  # это хозяйство подтормаживает и надо бы его рафакторить
    lm_t1_dw_base.visible = impl_trans and jaw_dw_base.visible
    lm_t2_dw_base.visible = not impl_trans and jaw_dw_base.visible
    lm_t1_up_base.visible = impl_trans and jaw_up_base.visible
    lm_t2_up_base.visible = not impl_trans and jaw_up_base.visible


# jaw_up_base.visible = 0  # for debug
# lm_up_base.visible = 0


def input(key):
    global impl_trans  # do transformation
    # upd_lm_visibility()
    # print(mouse.hovered_entity)
    if key == 'escape':
        quit()

    if key == 't':
        transform(impl_trans)
        impl_trans = not impl_trans

    if key == '1':
        jaw_up_base.visible = not jaw_up_base.visible
        lm_t1_up_base.visible = not lm_t1_up_base.visible
        lm_t2_up_base.visible = not lm_t2_up_base.visible

    if key == '2':
        jaw_dw_base.visible = not jaw_dw_base.visible
        lm_t1_dw_base.visible = not lm_t1_dw_base.visible
        lm_t2_dw_base.visible = not lm_t2_dw_base.visible


def update():
    # the whole jaw rotation/ movement (just for fun)
    if not held_keys['shift']:
        jaw_up[tooth_ind].z += (held_keys['s'] - held_keys['w']) * time.dt
        jaw_up[tooth_ind].y += (held_keys['a'] - held_keys['d']) * time.dt
        jaw_up[tooth_ind].x += (held_keys['z'] - held_keys['x']) * time.dt
    else:
        jaw_up[tooth_ind].rotation_z += (held_keys['s'] -
                                         held_keys['w']) * time.dt*10
        jaw_up[tooth_ind].rotation_y += (held_keys['a'] -
                                         held_keys['d']) * time.dt*10
        jaw_up[tooth_ind].rotation_x += (held_keys['z'] -
                                         held_keys['x']) * time.dt*10


# start running
app.run()
