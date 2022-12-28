''' конвертер csv файла в массивы для обучения сделано средствами pandas
    софт Ledas вытаскивающий лендмарки выдает csv файл со строками вида 
    Path,Jaw,Tooth_ID,Stage,MDWLine0_0,MDWLine0_1,MDWLine0_2,MDWLine1_0,MDWLine1_1,MDWLine1_2,BCPoint_0,BCPoint_1,BCPoint_2,FAPoint_0,FAPoint_1,FAPoint_2
    100310.oas,1,31,1,-0.61,-23.24,14.26,4.2,-22.5,14.24,1.77,-22.86,14.06,1.73,-24.16,10.44
    100310.oas,1,31,2,-0.78,-23.53,13.95,3.96,-22.42,14.06,1.57,-22.98,13.82,1.74,-24.26,10.19  
    4-е значение 1-T1, 2-T2    
    TODO - делать для обоих челюстей
'''
from copy import Error
from pprint import pp
import numpy as np
import pandas as pd
from time import time as t
import sys, math

#import torch
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt


def set_gen_fr_csv_pd(csv_path):
    # читает csv файл
    # выдает датасет - 2 объекта T1 и T2 длина каждого равна количеству уникальных кейсов
    up_teeth_nums = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28] # Jaw_id = 2 верхняя
    dw_teeth_nums = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38] # Jaw_id = 1 нижняя 

    df = pd.read_csv(csv_path, index_col='id')
    # уберем временно из датафрейма все верхние челюсти и оставим только нижние
    df = df[df['Jaw_id']==1]
    
    '''
    # отфильтруем дичь вида inf, конские значения, откуда там взявшиеся, интересно???
    ul = 100
    dl = -100
    # код ниже закоментирован потому что есть проблема - 
    # если в зубе дичь, удалять его надо целиком. т.е всю строку. 
    # как тут происходит -  я не уверен, надо проверять. ъ
    
    df = df.query(' StartT0_0   < @ul & StartT0_0   > @dl &\
                    StartT0_1   < @ul & StartT0_1   > @dl &\
                    StartT0_2   < @ul & StartT0_2   > @dl &\
                    EndT0_0     < @ul & EndT0_0     > @dl &\
                    EndT0_1     < @ul & EndT0_1     > @dl &\
                    EndT0_2     < @ul & EndT0_2     > @dl &\
                    StartT1_0   < @ul & StartT1_0   > @dl &\
                    StartT1_1   < @ul & StartT1_1   > @dl &\
                    StartT1_2   < @ul & StartT1_2   > @dl &\
                    EndT1_0     < @ul & EndT1_0     > @dl &\
                    EndT1_1     < @ul & EndT1_1     > @dl &\
                    EndT1_2     < @ul & EndT1_2     > @dl &\
                    BCPointT0_0 < @ul & BCPointT0_0 > @dl &\
                    BCPointT0_1 < @ul & BCPointT0_1 > @dl &\
                    BCPointT0_2 < @ul & BCPointT0_2 > @dl &\
                    FAPointT0_0 < @ul & FAPointT0_0 > @dl &\
                    FAPointT0_1 < @ul & FAPointT0_1 > @dl &\
                    FAPointT0_2 < @ul & FAPointT0_2 > @dl &\
                    BCPointT1_0 < @ul & BCPointT1_0 > @dl &\
                    BCPointT1_1 < @ul & BCPointT1_1 > @dl &\
                    BCPointT1_2 < @ul & BCPointT1_2 > @dl &\
                    FAPointT1_0 < @ul & FAPointT1_0 > @dl &\
                    FAPointT1_1 < @ul & FAPointT1_1 > @dl &\
                    FAPointT1_2 < @ul & FAPointT1_2 > @dl')
    '''                
    # список уникальных кейсов
    cases = df.Case_id.unique()
    # print("cases", cases)
    # print (f"len cases - {len(cases)} type {type(cases)}") # len cases - 2 type <class 'numpy.ndarray'>
    dataset_t0 = [[[0.00001 for j in range(12)] for tooth in range(16)] for k in range(len(cases))]# раньше было заполнено нулями 
    dataset_t1 = [[[0.00001 for j in range(12)] for tooth in range(16)] for k in range(len(cases))] # и тут тоже
    dataset_t0_case2 = [[0.00001 for j in range(12)] for tooth in range(16)] # для case2 ( для теста)
    dataset_t1_case2 = [[0.00001 for j in range(12)] for tooth in range(16)] 
    # перебираем по всем уникальным номерам ( по списку cases)
    for k, case in enumerate(cases):
        # перебираем по всем строкам из датафрейма, с одним и тем же Case_Id    
        subdf = df[df['Case_id']==case]
        for i in range(len(subdf)): # перебираем по части датафрейма где все Case_id = case
            # здесь - один row - это один зуб, надо заполнить челюсть значениями зубов
            row = subdf.iloc[i] # строка датафрейма
            row_ = row[3:] # в этой подстроке будем исткать невалидные конские значения т.к. номер зуба и Case_id могут быть большими
            if row_[row_>100].any() or row_[row_<-100].any(): # Если в строке есть числа <-100  или >100, такой зуб нахер
                continue

            row = row.tolist() # [7.0, 1.0, 36.0, 19.8454, 23.327, -1.1470 ... 982, -1.87889, 24.06, 27.2699, -3.97232, 0.0]
            tooth_id = int(row[2])&255 # 36 к примеру
            num_in_jaw = dw_teeth_nums.index(tooth_id)
            dataset_t0[k][num_in_jaw] = row[3:9] + row[15:21]
            dataset_t1[k][num_in_jaw] = row[9:15] + row[21:27]
            if (case == 2 ):
                dataset_t0_case2[num_in_jaw] = row[3:9] + row[15:21]
                dataset_t1_case2[num_in_jaw] = row[9:15] + row[21:27]      
    dataset_t0 = np.array(dataset_t0) # конверт их в numpy
    dataset_t1 = np.array(dataset_t1) 
    dataset_t0_case2 = np.array(dataset_t0_case2)
    dataset_t1_case2 = np.array(dataset_t1_case2)

    # dataset_t0[dataset_t0>100] = 0 # вариант обнулить все конские цифры. но не удалять зуб полностью. 
    # dataset_t0[dataset_t0<-100] = 0
    # dataset_t1[dataset_t1>100] = 0
    # dataset_t1[dataset_t1<-100] = 0
    
    return dataset_t0, dataset_t1, dataset_t0_case2, dataset_t1_case2

def set_gen_fr_csv_pd_ver2( csv_path,                   # путь к файлу csv
                            jaw_num,                    # номер челюсти 1 -нижняя, 2 верхняя, 3- обе 
                            missed_teeth_allowed=False, # разрешение на использование пропущенных зубов( пока не используется)
                            vector_len_one_jaw=14*15,   # (210) длина входного вектора значений для одной челюсти
                            prn=False):                 # вывод служебной инф. в консоль
    ''' Отличия новой версии файла - работа с новым набором лендмарок - 
        MDWLine (start, end), BCPoint, MeanRootApex, FEGJPoint
        вместо MDWLine (start, end), BCPoint, FAPoint
        длина вместо 168 на челюсть (12* 14) стала 210 (15*14) вторая версия файла, 
        функция берет только максимальный номер стейджа 
        выдает датасет - 2 объекта T1 и T2 длина каждого равна количеству уникальных кейсов
        учить будем только кейсами где присутствуют 28 зубов 2 челюсти без третьих моляров.
        челюсти выдаются либо одна, либо две см. jaw_num 
        На выход попадают только кейсы где есть все зубы, без пропусков, исключая третьи моляры (восьмерки)
    '''
    assert(jaw_num in {1,2,3})
    vector_len = vector_len_one_jaw if jaw_num in (1,2) else vector_len_one_jaw * 2 # длина вектора лендмарок (размер входного слоя)
    ts = t() # time stamp for spend time calc
    up_teeth_nums = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27] # Jaw_id = 2 верхняя / по 14 зубов 
    dw_teeth_nums = [37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47] # Jaw_id = 1 нижняя 
    cases_with_missed_tt = [] # для сбора NN кейсов с отсутствующими зубами

    df = pd.read_csv(csv_path, index_col='Path')

    out_dataset =[] # выходной датасет вида [ (челюстьT1, челюстьT2) .... ]
    #df_cases = [df[(df.index == case)] for case in df.index.unique()] быстрее не делает
    #df_unstaged = [df_case[df_case.Stage == df_case.Stage.max()] for df_case in df_cases]
    for case in df.index.unique(): # cycle for unique cases
        if 1:
            stop_case = False    # флаг, если в кейсе не все зубы и кейс г, переход к следующему
            # подвыборка с одинаковыми case
            df_case = df[(df.index == case)]
            # # проверяем case на наличие в нем какой-нибудь дичи типа inf 
            # # такой кейс не рассматриваем, переходим к следующему
            # if df_case[df_case>100].any() or df_case[df_case<100].any():
            #     cases_with_missed_tt.append(case)
            #     continue            

            # далее надо сделать так чтобы для одного зуба и одной челюсти существовал кортеж в котором первый член будет обучающий Т1, а второй Т2
            df_case_t1 = df_case[df_case.Stage == 1]
            df_case_t2 = df_case[df_case.Stage == 2]
            # все цифры из df надо вытащить в линейку 
            # при этом если какого-то зуба нет, такой кейс нам не годится. 
            # итерируемся по 14(28 - для двух челюстей) и  
            case_out_t1 =[] # 168*2 значения все лендмарки - всех зубов челюсти для Т1
            case_out_t2 =[] # то же для Т2
            
            # for i in range(14):# len(df_case_t1.index)): нет, не по индексу, а именно по 14 для одной челюсти - 

            if jaw_num ==1:
                teeth_nums = dw_teeth_nums
            elif jaw_num ==2:
                teeth_nums = up_teeth_nums
            else:
                teeth_nums = dw_teeth_nums + up_teeth_nums
                    
            for Tooth_ID in teeth_nums:
                # !!!!!!!!!!! тут надо набивать не абы как а в соответствии с тем как идут зубы дизайнере - сначала 31, потом 32, .. 37, 
                # в челюсти должны быть все, кроме третьих моляров
                # надо проверять есть ли зуб с данным tooth_id в подвыборке для этого кейса
                if (not Tooth_ID in df_case_t1.Tooth_ID.values):
                    # и если не находим такого зуба в челюсти то эту челюсть в выборку не добавляем, 
                    # возможно проверки излишняя, возможно нет. надо проверять что выдает софт Ледас
                    cases_with_missed_tt.append(case)
                    stop_case = True
                    break
                
                # row - Jaw  Tooth_ID  Stage  MDWLine0_0  MDWLine0_1  MDWLine0_2  MDWLine1_0  MDWLine1_1  MDWLine1_2  BCPoint_0  BCPoint_1  BCPoint_2  FAPoint_0  FAPoint_1  FAPoint_2
                # row = row.tolist() # [7.0, 1.0, 36.0, 19.8454, 23.327, -1.1470 ... 982, -1.87889, 24.06, 27.2699, -3.97232, 0.0]
                row_t1 = df_case_t1[df_case_t1.Tooth_ID == Tooth_ID]
                # row_t2 = df_case_t2.iloc[i]
                #try:
                row_t2 = df_case_t2[df_case_t1.Tooth_ID == Tooth_ID]
                #except:
                #    print(f"STOP  {case}")

                # if row_t1[row_t1>100].any():
                #     print(f"ДИЧЬ!!! case {case}")
                #     sys.exit()

                # tooth_id = dw_teeth_nums[i] # tooth_id = int(row[1])&255 # 36 к примеру
                # print(f"df_case_t1.Tooth_ID.values {df_case_t1.Tooth_ID.values}") 28 штук
                # далее надо сделать челюсть. 
                case_out_t1+=row_t1.values.tolist()[0][3:]
                case_out_t2+=row_t2.values.tolist()[0][3:]
            
            if stop_case == True: 
                continue
            
            # перевод в np необходим для работы торча впоследствии
            case_out_t1 = np.float32(case_out_t1)
            case_out_t2 = np.float32(case_out_t2)

            # прежде чем добавлять полученные строки в датафрейм, 
            # нужно проверить их на наличие дичи типа inf и прочее. 
            if (   case_out_t1[case_out_t1>100].any()
                or case_out_t1[case_out_t1<-100].any()
                or case_out_t2[case_out_t1>100].any()
                or case_out_t2[case_out_t1<-100].any()):
                stop_case = True
        
            if stop_case == True: 
                continue
            
            assert(len(case_out_t1) == len(case_out_t2) == vector_len) # проверяем что все зубы на месте, 
            
            out_dataset.append((case_out_t1, case_out_t2)) # аппендим кортеж из двух векторов, в каждом по 168*2 значения
        #except:
        #    print(f"case {case}")
    if prn:
        print(f"length of cases_with_missed_tt - {len(cases_with_missed_tt)} - {cases_with_missed_tt[:3]} ... ")
        print(f"dataset length =  {len(out_dataset)} cases")
        print(f"spend  {t()-ts:.1f} sec")

    # вывод кортежа из пар - трейн(T1) и метка (T2)
    # делить на трейн и тест можно потом 
    # out [([<168>T1], [<168>T2]), <next case tuple>, <next...>, ...]
    return out_dataset 

def set_gen_fr_csv_pd_ver3( csv_path,                   # путь к файлу csv
                            jaw_num=3,                  # номер челюсти 1 -нижняя, 2 верхняя, 3- обе 
                            missed_teeth_allowed=False, # разрешение на использование пропущенных зубов( пока не используется)
                            vector_len_one_jaw=16*15,   # (16*15=240) длина входного вектора значений для одной челюсти
                            prn=False,                  # вывод служебной инф. в консоль
                            one_case=""):               # используется для выдачи только одного кейса для теста конкретного файла.
    ''' Попытка сделать выбор с нулевыми лендмарками.
        Работает с новым набором лендмарок - 
        MDWLine (start, end), BCPoint, MeanRootApex, FEGJPoint
        вместо MDWLine (start, end), BCPoint, FAPoint
        длина вместо 210 на челюсть (12* 14) стала 240 (16*15) вторая версия файла, 
        функция берет только максимальный номер стейджа 
        выдает датасет - 2 объекта T1 и T2 длина каждого равна количеству уникальных кейсов
        учить будем только кейсами где присутствуют 32 зуба в 2 челюстях.
        челюсти выдаются либо одна, либо две см. jaw_num 
        
    '''
    assert(jaw_num in {1,2,3})
    vector_len = vector_len_one_jaw if jaw_num in (1,2) else vector_len_one_jaw * 2 # длина вектора лендмарок (размер входного слоя)
    ts = t() # time stamp for spend time calc
    up_teeth_nums14 = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27] # Jaw_id = 2 верхняя / по 14 зубов 
    dw_teeth_nums14 = [37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47] # Jaw_id = 1 нижняя 
    up_teeth_nums16 = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28] # Jaw_id = 2 верхняя / по 16 зубов 
    dw_teeth_nums16 = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48] # Jaw_id = 1 нижняя  / 16 зубов 
    cases_with_missed_tt = [] # для сбора NN кейсов с отсутствующими зубами

    df = pd.read_csv(csv_path, index_col='Path')

    out_dataset =[] # выходной датасет вида [ (челюстьT1, челюстьT2) .... ]
    #df_cases = [df[(df.index == case)] for case in df.index.unique()] быстрее не делает
    #df_unstaged = [df_case[df_case.Stage == df_case.Stage.max()] for df_case in df_cases]
    indxs = df.index.unique()
    print(f"start collecting data")
    for indx, case in enumerate(df.index.unique()): # cycle for unique cases
        
        # режим работы с выдачей только одного кейса
        if one_case != "":
            if case != one_case : continue
        
        # просто печать точек чтоб видеть что процесс жив.     
        if (indx+1)%10==0 : 
            print(".", end="") # ...............
        if (indx+1)%1000==0 : print()
        
        # подвыборка с одинаковыми case
        df_case = df[(df.index == case)]
        steps = {1:(1,), 2:(2,), 3:(1,2)} # 1 раз ходить по циклу для одной ч-сти, 2 - для двух. 

        # stop_case = False    # флаг, если в кейсе не все зубы и кейс г, переход к следующему
        # далее надо сделать так чтобы для одной (ДВУХ, ЕСЛИ jaw_num=3) челюсти существовал кортеж в котором 
        # первый член будет обучающий Т1, а второй Т2
        df_case_t1 = df_case[df_case.Stage == 1]
        df_case_t2 = df_case[df_case.Stage == 2]
        
        # print(f"\ndf_case_t1\n {df_case_t1} \n length- {len(df_case_t1)}")
        # print(f"\ndf_case_t2\n {df_case_t2} \n length- {len(df_case_t2)}")
        
        # все цифры из df надо вытащить в линейку 
        # при этом если какого-то зуба нет, такой кейс нам не годится. 
        # итерируемся по 16(32 - для двух челюстей) и  
        case_out_t1 = [] # 168*2 значения все лендмарки - всех зубов челюсти (ИЛИ ДВУХ ЧЕЛЮСТЕЙ) для Т1
        case_out_t2 = [] # # то же для Т2
        
        for i in steps[jaw_num]:
            case_out_t1_1_step = [] # это для одного шага
            case_out_t2_1_step = [] # то же для Т2
            
            assert i in (1,2)
            teeth_nums = dw_teeth_nums16 if i == 1 else up_teeth_nums16
                  
            for Tooth_ID in teeth_nums:
                # !!!!!!!!!!! тут надо набивать не абы как а в соответствии с тем как идут зубы дизайнере - сначала 31, потом 32, .. 37, 
                # надо проверять есть ли зуб с данным tooth_id в подвыборке для этого кейса
                if (not Tooth_ID in df_case_t1.Tooth_ID.values):
                    # и если не находим такого зуба в челюсти то эту челюсть запоминаем и продолжаем со следующим зубом
                    if not case in cases_with_missed_tt: # проверять тут на существование нужно, т.к. если челюсти 2, 
                        # и в обоих есть пропуски, то кейс задублируется, а это не надо. 
                        cases_with_missed_tt.append(case)
                    #stop_case = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOTE
                    #break               #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOTE
                    # раньше тут брякали, сейчас продолжаем набивать кейс
                    # если какого-то зуба нет добавляем следующий по списку. 
                    continue
                
                # row - Jaw  Tooth_ID  Stage  MDWLine0_0  MDWLine0_1  MDWLine0_2  MDWLine1_0  MDWLine1_1  MDWLine1_2  BCPoint_0  BCPoint_1  BCPoint_2  FAPoint_0  FAPoint_1  FAPoint_2
                # row = row.tolist() # [7.0, 1.0, 36.0, 19.8454, 23.327, -1.1470 ... 982, -1.87889, 24.06, 27.2699, -3.97232, 0.0]
                row_t1 = df_case_t1[df_case_t1.Tooth_ID == Tooth_ID]
                row_t2 = df_case_t2[df_case_t1.Tooth_ID == Tooth_ID]
                # далее надо сделать челюсть. 
                try:
                    case_out_t1_1_step+=row_t1.values.tolist()[0][3:] # добавляем в список по 15 значений от каждого зуба. 
                    case_out_t2_1_step+=row_t2.values.tolist()[0][3:]
                except:
                    # print(f"\ndf_case_t1\n {df_case_t1} \n length- {len(df_case_t1)}")
                    # print(f"\ndf_case_t2\n {df_case_t2} \n length- {len(df_case_t2)}")
                    # sys.exit()
                    pass
            # недостающие зубы заменяем нулями
            while len(case_out_t1_1_step) < vector_len_one_jaw:
                case_out_t1_1_step+=[0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
            while len(case_out_t2_1_step) < vector_len_one_jaw:
                case_out_t2_1_step+=[0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
            
            case_out_t1 += case_out_t1_1_step # если шагов 2, то эта балалайка удвоится
            case_out_t2 += case_out_t2_1_step # и эта тоже

            #if stop_case == True: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOTE
            #    continue
            
        # перевод в np необходим для работы торча впоследствии
        case_out_t1 = np.float32(case_out_t1)
        case_out_t2 = np.float32(case_out_t2)

        # прежде чем добавлять полученные строки в датафрейм, 
        # нужно проверить их на наличие дичи типа inf и слишком больших чисел. 
        limit_val = 200
        if (   case_out_t1[case_out_t1>limit_val].any()
            or case_out_t1[case_out_t1<-limit_val].any()
            or case_out_t2[case_out_t1>limit_val].any()
            or case_out_t2[case_out_t1<-limit_val].any()):
            #stop_case = True
            # print(f" BIG VALUE!!! case_out_t1 {case_out_t1}")
            # если есть недопустимые значения, пропускаем кейс.
            continue
    
        #if stop_case == True: 
        #    continue
        if 0: #case == '100090.oas':
            print(f"asdf {case}")
            print(f" ")
        assert(len(case_out_t1) == len(case_out_t2) == vector_len) # проверяем что длина вектора правильная. иначе в торч это не полезет.
        
        out_dataset.append((case_out_t1, case_out_t2)) # аппендим кортеж из двух векторов, в каждом по 210 значений

    if prn:
        print(f"\nlength of cases_with_missed_tt - {len(cases_with_missed_tt)} - {cases_with_missed_tt[:3]} ... ")
        print(f"dataset length =  {len(out_dataset)} cases")
        print(f"spend  {t()-ts:.1f} sec")

    # вывод кортежа из пар - трейн(T1) и метка (T2)
    # делить на трейн и тест можно потом 
    # out [([<168>T1], [<168>T2]), <next case tuple>, <next...>, ...]
    return out_dataset 

if __name__ == "__main__":
    # fpath = 'C:\\Projects\\jaw_encoder\\csv\\input.csv' 
    fpath = r"C:\my\csv_test\test_for_pd.csv"
    fpath = r"C:\Projects\Spark\orthoplatform\Source\Scripts\BatchTesting\ou3\8k.csv" # 5k кейсов. 
    fpath = r'C:\Projects\torchEncoder\csv\11k.csv'
    fpath = r"C:\Projects\torchEncoder\csv\Diego_1k.csv"
    # t0, t1, t0_case2, t1_case2 = set_gen_fr_csv_pd(fpath)
    # t0, t1, t0_case2, t1_case2 = 
    
    ''' это тесты 
    x = ('1', '2', '3', '4') # строки останутся строками
    x= (11,22,33,44) # это будут тензоры 
    y = (1,2,3,4) # числа - всегда тензор 
    ds = tuple(zip(x,y))
    нах тут не нужен торч. хватит и массива. торч можно запилить в трене.
    loader = DataLoader(ds, batch_size=1)
    print(f"loader info {loader.dataset}")
    for x,y in loader:
        print(f"x {x}")
        print(f"y {y}")
    '''

    dataset = set_gen_fr_csv_pd_ver3(fpath, prn=True)
    # print(f"dataset\n {dataset}")

    # landmarks_fr_file = set_gen_fr_csv_pd_ver3(r"C:\Projects\torchEncoder\csv\Diego_1k.csv", 
        # one_case="123957.oas")
    # print(f"dataset\n {landmarks_fr_file}")
    # print(f"length out_dataset {len(dataset)} -> {len(dataset[0][0])} ")

    # print (f"t0[t0>100] {t0[t0>100]}")
    # assert len(t0[t0>100]) == 0 # если нет значений выше порога, должен возвращать пустой массив
    # assert len(t0[t0<-100]) == 0
    # assert len(t1[t1>100]) == 0
    # assert len(t1[t1<-100]) == 0
    #print (f"t0 {t0[2]} shape {t0.shape} len {len(t0)}") # shape(?, 16, 12)

