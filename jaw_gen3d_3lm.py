# новый генератор с 3-мя лендмарками 
# UPD mdw point2 вместо длины mdw

# import cv2
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# класс который генерит картинки размерностью с искривлениями зубов или без.
# основное отличие - генерятся не картинки с зубами, а данные в виде 
# [[метка][координаты точки начала, длина вектора]]
# вектора в этой версии 3-х мерные, поэтому без opencv в этот раз

np.random.seed() # устанавливает режим случайных чисел без повторений от запуска к запуску

class Landmark_gen():
    ''' creates pictures with MDW, BCP, FAP landmarks '''

    def __init__(self, dim=(200, 200, 200)):
        self.dim = dim

    def image_gen(  self, 
                    name='_', 
                    scale=1, 
                    factor=2.6, 
                    spoiledMDW = False,     # вводит хаотичный наклон и смещение зубов имитируя T1 для лендмарка MDW
                    shiftMDW = False        # add random shift for whole tooth for X,Y (Z?) axis для лендмарка MDW
                ):
        # возвращает 2 вектора - ровных и кореженных зубов 
        # все делаем под разрешение 200x200 
        self.name= name
        # self.back = np.zeros(self.dim)            # картинка с ровными зубами
        # self.back_spoiled = np.zeros(self.dim)     # картинка с корявыми зубами
        self.factor = factor
        self.out_vector = []                        # вектор лендмарков [x,y,len_x, len_y]
        self.out_vector_spoiled = []                # то же, но искривленные зубы
        center = (self.dim[0]//2, self.dim[1]//2, self.dim[2]//2) # центральная точка на графике
        
        # Генерим Medial Distal Width Line лендмарк
        # строим дугу. так чтобы на основе ее точек можно было зубья подровнять
        # сначала на плоскости, потом по мере добавим изменения координат по оси z
        # инициируем все точки лендмарков 
        point_mdw, point2_mdw, len_x, len_y, len_z, point_bcp, point_fap = [0,0,0], [0,0,0], 0,0,0, [0,0,0], [0,0,0]
        # то же для спойленых точек
        point_mdw_s, point2_mdw_s, len_x_s, len_y_s, len_z_s, point_bcp_s, point_fap_s = [0,0,0], [0,0,0], 0,0,0, [0,0,0], [0,0,0]

        for i in range(-8, 8): # от зуба к зубу
            
            #### Формируем лендмарк Medial Distal Width line MDW #### 
            
            # в вектор добавляются 3 значения - координаты начальной точки x,y,z,
            # + 3 значения проекций отрезка на оси, всего 6 значений
            # остальные лендмарки просто будут добавляться в хвост.
            
            # строим точки графика 
            z_lev = center[2]         # уровень на котором будет плоскость "окклюзии" в координате z
            # первая точка зуба
            point_mdw = [int(i*scale)+center[0], int(self.duga(i)) + center[1], z_lev] 
            # вторая
            point2_mdw = [int((i+1-0.2)*scale +
                          center[0]), int(self.duga(i+1-0.2)) + center[1], z_lev]
            
            # строим линии по дуге - ровные зубы
            # cv2.line(self.back, point_mdw, point2_mdw, 1, 2)
            # добавляем координаты точек в выдачу
            len_x = point2_mdw[0] - point_mdw[0]
            len_y = point2_mdw[1] - point_mdw[1]
            len_z = 0
            # это теперь будем добавлять в самом конце
            # self.out_vector.append([point_mdw[0], point_mdw[1], point_mdw[2], len_x, len_y, len_z])

            # шатаем зубы если задано шатать и выдаем на картинку back_spoiled
            if spoiledMDW: 
                if i in [-8, -7, -6, 5, 6, 7 ]: # портим моляры 
                    point_mdw_s[0] = point_mdw[0] + int((np.random.sample()-1)*i)

                if i in [-5, -4, -3, -2, 1, 2, 3, 4]: # премоляры и резцы
                    point_mdw_s[1] = point_mdw[1] + int((np.random.sample()-1)*i*2)

                if i in [-1, 0]: # передние резцы
                    point_mdw_s[1] = point_mdw[1] + int((np.random.sample()-1)*4)
            
            if shiftMDW:
                case_shift_x_ = int((np.random.sample()-1)*5)
                point_mdw_s[0] = point_mdw[0] +  case_shift_x_
                point2_mdw_s[0] = point2_mdw[0] +  case_shift_x_ # объединить сдвиг по x и y в один точку 2 убрать, ширину не менять, см.ниже            
                case_shift_y_ = int((np.random.sample()-1)*5)
                point_mdw_s[1] = point_mdw[1] + case_shift_y_
                point2_mdw_s[1] = point2_mdw[1] + case_shift_y_
            
            point_mdw_s[2] = z_lev
            
            # для второго вектора делаем покореженные значения. или те-же если корежить не надо
            len_x_s = len_x # point2_mdw_s[0] - point_mdw_s[0]
            len_y_s = len_y # point2_mdw_s[1] - point_mdw_s[1] ##### !!!! ??? а не нужно ли здесь ширину оставить как была? она не меняется так-то
            # добавляем координаты точек в выдачу, но в конце 
            # self.out_vector_spoiled.append([point_mdw_s[0], point_mdw_s[1], point_mdw_s[2], len_x_s, len_y_s, len_z_s])
        
            #### Формируем лендмарк Buccal Cusp Points BCP ####

            # в вектор добавляем 3 значения - координаты точки
            # он повторяет точки  MDW с небольшим сдвигом верх по z и вверх по х
            point_bcp = [point_mdw[0] - 5, point_mdw[1], point_mdw[2]+4]
            # добавим небольшой шум к bcp
            point_bcp_s = [val + int((np.random.sample()-1)*2) for val in point_bcp]

            #### Формируем лендмарк Front Acess Point FAP ####

            # он повторяет точки  MDW с небольшим сдвигом вниз по z 
            point_fap = [point_mdw[0], point_mdw[1], point_mdw[2]-9]
            # добавим небольшой шум к bcp
            point_fap_s = [val + int((np.random.sample()-1)*4) for val in point_fap]
    
            # теперь набиваем вектора созданными значениями
            self.out_vector.append(
                                   [point_mdw[0],   point_mdw[1],   point_mdw[2], #len_x, len_y, len_z, 
                                    point2_mdw[0],  point2_mdw[1],  point2_mdw[2],
                                    point_bcp[0],   point_bcp[1],   point_bcp[2], 
                                    point_fap[0],   point_fap[1],   point_fap[2]                
                                ])

            self.out_vector_spoiled.append(
                                   [point_mdw_s[0],     point_mdw_s[1],     point_mdw_s[2], #len_x_s, len_y_s, len_z_s, 
                                    point2_mdw_s[0],    point2_mdw_s[1],    point2_mdw_s[2],
                                    point_bcp_s[0],     point_bcp_s[1],     point_bcp_s[2], 
                                    point_fap_s[0],     point_fap_s[1],     point_fap_s[2]                
                                ])
        # print (f"vector {self.out_vector}")
        # print (f"vector_spoiled {self.out_vector_spoiled}")
        return self.out_vector, self.out_vector_spoiled  


    def duga(self, x):
        ''' returns arc function'''
        return abs(x)**self.factor - self.dim[0] * 0.35

    def draw_pic_fr_vec(self, vec):
        # это осталось из версии 2D возможно придется удалить
        # get landmarks and Draw pictures
        # vec это вектор лендмарка длиной 16, vec[i] = [x,y, x_len, y_len]
        self.pred_pict = np.zeros(self.dim)   
        for i in range(len(vec)):
            x,y, x_len, y_len = vec[i]
            # cv2.line(self.pred_pict, (int(x), int(y)), 
                # (int(x+x_len), int(y+y_len)), 1, 2)
        
        return self.pred_pict
    
    def draw_3d(self, *args,  style='v', show=False):
        # draw 3d pictures in matplotlib
        # print(f" {args}")
        assert(style in ('h', 'v'))
        MDWcolors = ['b','r','g','magenta']
        BCPcolors = ['yellow', 'yellow', 'yellow']
        FAPcolors = ['k', 'k', 'k']
        titls2 = ["T1", "Predicted T2"]
        titls3 = [ "T1", "T2", "Predicted T2",""]
        # mpl.rcParams['legend.fontsize'] = 10
        # n = min([x.shape[0] for x in args]) # сколько столбцов нарисуем
        n = min([len(x) for x in args]) # сколько столбцов нарисуем
        
        # горизонтальное или вертикальное нарисование
        if style == 'v':
            rows = len(args)
            cols = n
        elif style == 'h':
            rows = n
            cols = len(args)
        else:
            """ тут надо обработать исключение """

        # fig = plt.figure('AutoSetup ML', figsize=(7*cols, 7*rows)) # with Title
        fig = plt.figure(figsize=(7*cols, 7*rows)) # with Title
        # fig.canvas.set_window_title('AutoSetup ML') deprecated 
        # так для каждого вызова в цикле открывается новое окно
        fig.canvas.manager.set_window_title('AutoSetup ML') 
        
        for j in range(n):
            for i in range(len(args)):
                if len(args) == 2:
                    titls = titls2  
                elif len(args) == 1:
                    titls = ['Jaw', '', '']
                else:
                    titls = titls3
                
                indx = i*n + j + 1 if style == 'v' else j*len(args) + i + 1
                ax = plt.subplot(rows, cols, indx, title = f'{titls[i]} {j}', projection='3d')  
                uu = args[i]
                q= args[i][j] # промежуточная ня чтобы проверить что там челюсть а не труля-ля ля
                q= np.array(q) # для универсальности, т.к. аргументы могут и не быть массивами np

                for t in q: # q = 16 зубов * [x,y,z, len_x, len_y, len_z, bcp_x, bcp_y, bcp_z, fap_x, fap_y, fap_z]
                    # рисоваем MDW но не рисуем нули от удаленных зубов.
                    # Причем лучше оценивать 
                    tr1 = 0.8 # thresh for mdw нужно сильно подбирать, т.к. много мусора.
                    tr2 = 0.7 # thresh for bcp
                    tr3 = 0.8 # thresh for fap
                    # притаскиваем все что близко к нулю, точно к нулю. 
                    # t[abs(t)<0.01] =0 
                    # и если нет нулей то печатаем лендмарк. 
                    if abs(t[0])<tr1 and abs(t[1])<tr1 and abs(t[2])<tr1 and abs(t[3])<tr1 and abs(t[4])<tr1 and abs(t[5])<tr1:
                    # if not 0 in t: # это пока не очень хорошо работает, надо доделать.
                        pass # не печатаем 
                        # print (f"пропуск {t}")
                    else:
                        ax.plot(
                        [t[0],t[3]], 
                        [t[1],t[4]], 
                        [t[2],t[5]], 
                        # label='норм', title=f"{i,j}", color = colors[i])
                        label='норм', color = MDWcolors[i])
                    
                    # t[abs(t)<0.05] =0  # еще разок для точечных лендмарков с более грубой притяжкой.
                    # точки bcp
                    if abs(t[6])<tr2 and abs(t[7])<tr2 and abs(t[8])<tr2:
                        pass
                        # print (f"ex BCP \n{t}")
                    else:
                        ax.scatter3D(t[6], t[7], t[8], color =BCPcolors[i])

                    # и также fap
                    if abs(t[9])<tr3 and abs(t[10])<tr3 and abs(t[11])<tr3:
                        pass # почему так, а потому что так короче писать. 
                        # print (f"ex FAP \n{t}")
                    else:
                        ax.scatter3D(t[9], t[10], t[11], color =FAPcolors[i])
                        
                    
                # ax.title.set_text('My')
                lim_ax_x_y = 35
                ax.set_xlim(-lim_ax_x_y,    lim_ax_x_y)
                ax.set_ylim(-30,    30)
                ax.set_zlim(-20,    20)
                ax.view_init(90,   90)

        plt.show() if show == True else None
        

if __name__ == "__main__":
    ex = Landmark_gen() 
    # генерируем разные картинки
    m1 = []
    m2 = []
    for i in range(2):  
        scale =  9 + np.random.sample()*3    # 9,3 это для размера 200, для размера 400 - диапазон от 14 до 19 
        factor = 2.2 + np.random.sample()/4   # 2.2, 4 - это для размера 200, для размера 400 - диапазон 2.5 - 2.7  
        name = f"scale {scale:.4}  factor {factor:.3}"
        vec = ex.image_gen(scale=scale, factor=factor, name=name, spoiledMDW=True, shiftMDW=True)
         
        m1.append(vec[0])
        m2.append(vec[1])
        # print (f"vec shape{vec[0][0]}")
        # print (f"vec[0][0]{vec[0][0]}")
    # NOTE - векторов m1 и m2 длины 1 h стить работает корректно, для других длин нет. 
    # v стиль работает корректно для всех длин 
    ex.draw_3d(m1, m2, show=True, style='h')


