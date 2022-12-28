""" Tool for visualize the ML model inference, staging process animation
"""
from ursina import *
from csv_parser_pd_ver2 import set_gen_fr_csv_pd_ver3
import cv2, numpy as np

# create a window
app = Ursina(development_mode=True, borderless=False)

# debug camera
cam = EditorCamera(rotation=(-90, 0, 0))

# light
L1 = PointLight(parent=cam, x=0, y=0, z=-16, color=color.rgba(80, 80, 80))

mesh_paths = [
    [f"../stl/123957/ToothSurface_{q}{n}.stl" for n in range(1, 9)] for q in range(1, 5)]
# Entityes
jaw_up_base = Entity(scale=(.1, .1, -.1), flipped_faces=True, collider="box") # collider???
jaw_dw_base = Entity(scale=(.1, .1, -.1), flipped_faces=True, collider="box")

jaw_up = [[Entity(parent=jaw_up_base, model=tooth) for tooth in q]
          for q in mesh_paths[:2]]
jaw_up = [[Entity(parent=jaw_dw_base, model=tooth) for tooth in q]
          for q in mesh_paths[2:]]

# draw landmarks
csv_path = r"C:\Projects\torchEncoder\csv\diego_landmarks_over_900.csv"
landmarks = set_gen_fr_csv_pd_ver3(csv_path, one_case="123957.oas")[0]
t1_landmarks = landmarks[0]
t2_landmarks = landmarks[1]

t1_landmarks = t1_landmarks.reshape(-1, 3)  # group by 3
t2_landmarks = t2_landmarks.reshape(-1, 3)  # group by 3
# print("len", t1_landmarks)


# draw landmarks lower jaw first
# base landmark need for control all batch
lm_down_base = Entity(scale=(0.2, 0.2, -.2),)
lm_down = [Entity(parent=lm_down_base, model='sphere', 
                flipped_faces=False,
                color=color.lime,
                scale=(.5, .5, .5),  # affects only zoom
                position=(lm[0]/2, lm[1]/2, lm[2]/2),
                ) for lm in t1_landmarks[:240//3]]
# then upper``
lm_up_base = Entity(scale=(0.2, 0.2, -.2),)
lm_up = [Entity(parent=lm_up_base, model='sphere',
                flipped_faces=False,
                color=color.red,
                scale=(.5, .5, .5),  # affects only zoom
                position=(lm[0]/2, lm[1]/2, lm[2]/2),
                ) for lm in t1_landmarks[240//3:]]

# transform calc np.float32(pts1)
# transform = cv2.estimateRigidTransform(t1_landmarks, t2_landmarks, fullAffine = False)
ret, M, mask = cv2.estimateAffine3D(np.float32(t1_landmarks),
                                    np.float32(t2_landmarks),
                                    # confidence = .9999999)
                                    confidence = .99)
assert ret, 'Transform calculation failed..'
# print(f"M {M}")
# print(f"mask {mask.shape}", end=" ")

def input(key):
    # global teeth
    if key == 'escape':
        quit()

    if key == '1':
        jaw_up_base.visible = False if jaw_up_base.visible == True else True
        lm_up_base.visible = False if lm_up_base.visible == True else True

    if key == '2':
        jaw_dw_base.visible = False if jaw_dw_base.visible == True else True
        lm_down_base.visible = False if lm_down_base.visible == True else True
    # print(f"jaw_dw.position {jaw_dw.position}")
    # print(f"jaw_dw.rotation {jaw_dw.rotation}")


def update():
    if held_keys['shift']:
        jaw_dw_base.z += (held_keys['s'] - held_keys['w']) * time.dt
        jaw_dw_base.y += (held_keys['a'] - held_keys['d']) * time.dt
        jaw_dw_base.x += (held_keys['z'] - held_keys['x']) * time.dt
    else:
        jaw_dw_base.rotation_z += (held_keys['s'] - held_keys['w']) * time.dt
        jaw_dw_base.rotation_y += (held_keys['a'] - held_keys['d']) * time.dt
        jaw_dw_base.rotation_x += (held_keys['z'] - held_keys['x']) * time.dt


# start running
app.run()
