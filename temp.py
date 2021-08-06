# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:54:43 2021

@author: zongsing.huang
"""

import pandas as pd
import numpy as np
import time

def fixed_round(var):
    if var%1>=.5:
        return np.ceil(var)
    else:
        return var//1

def task_priority(total_number_of_tasks):
    n = total_number_of_tasks
    v1 = np.zeros(n)
    
    for i in range(n):
        v1[i] = np.random.uniform(low=i, high=i+1)
    
    for i in range(int(n/2)):
        l, j = np.random.choice(np.arange(n), size=2, replace=False)
        v1[l], v1[j] = v1[j], v1[l]
    
    return v1

def worker_allocation(total_number_of_stations):
    m = total_number_of_stations
    v2 = np.zeros(m)
    
    v2 = np.arange(m)
    np.random.shuffle(v2)
    
    return v2

def task_sequence(total_number_of_tasks, task_priority, set_of_predecessors):
    n = total_number_of_tasks
    v1 = task_priority
    predecessors = set_of_predecessors
    
    v3 = []
    A = [i for i in range(n) if predecessors[i]==[]]
    while len(A)>0:
        j = A[np.argmax(v1[A])]
        v3.append(j)
        A.remove(j)
        
        for task, Pre_j in enumerate(predecessors):
            if task not in v3 and task not in A  and set(Pre_j).intersection(set(v3))==set(Pre_j) and set(Pre_j)!=set():
                A.append(task)
                
    v3 = np.array(v3)
    if 0 not in v3 or 1 not in v3 or 2 not in v3 or 3 not in v3 or 4 not in v3 or 5 not in v3 or 6 not in v3 or 7 not in v3 or 8 not in v3 or 9 not in v3:
        print('task_sequence error')
        print(v3)
        print('='*20)
    return v3

def breakpoint_vector(worker_allocation, task_sequence,
                      processing_time):
    m = len(worker_allocation)
    n = len(task_sequence)
    v2 = worker_allocation
    v3 = task_sequence

    Clb = fixed_round(np.min(task_time_unit, axis=1).sum() / total_number_of_stations)
    Cub = fixed_round(np.max(task_time_unit, axis=1).sum() / total_number_of_stations + np.max(task_time_unit))

    t = processing_time
    v4 = [0, ]
    
    while (Cub-Clb>1):
        Ct = fixed_round( (Clb+Cub) / 2 )
        i = 0
        Wt = 0
        Si = []
        v4 = [0, ]
        
        for j in range(n):
            k = v3[j].astype(int) # 任務
            w = v2[i].astype(int) # 工人
            
            if Wt+t[k, w]>Ct:
                v4.append(v4[-1] + len(Si))
                # print('工人'+str(w+1)+'執行任務'+str(k+1)+'，成本'+str(t[k, w])+'，累積成本'+str(Wt+t[k, w])+'，大於限制成本'+str(Ct))
                # print('工人'+str(w+1)+'應執行任務為'+str([x+1 for x in Si]))
                i = i + 1
                Si = [k]
                if i>=m:
                    break
                w = v2[i].astype(int)
                Wt = t[k, w]
            else:
                w = v2[i].astype(int)
                Wt = Wt + t[k, w]
                Si.append(k)
                # print('工人'+str(w+1)+'執行任務'+str(k+1)+'，成本'+str(t[k, w])+'，累積成本'+str(Wt)+'，小於限制成本'+str(Ct))
            
        v4.append(10)
        
        if i<=m-1:
            Cub = Ct
        else:
            Clb = Ct
    
    v4 = np.array(v4)
    # 因為不是每次最後結果都是5，所以可以嘗試儲存最後一個成功案例當輸出
    # 或者就是失敗就不覆蓋
    return v4

def AX_for_task(task_priority, pc):
    popSize = task_priority.shape[0]
    task_priority1 = task_priority[:int(popSize/2)]
    task_priority2 = task_priority[int(popSize/2):]
    new_x1 = np.zeros_like(task_priority1)
    new_x2 = np.zeros_like(task_priority2)
    
    for i in range(int(popSize/2)):
        r = r = np.random.uniform()
        x1 = task_priority1[i]
        x2 = task_priority2[i]
        if r<=pc:
            lam1 = np.random.uniform()
            lam2 = 1 - lam1
            
            new_x1[i] = lam1*x1 + lam2*x2
            new_x2[i] = lam1*x2 + lam2*x1
        else:
            new_x1[i] = x1.copy()
            new_x2[i] = x2.copy()
    
    new_x = np.vstack([new_x1, new_x2])
    return new_x

def WMX_for_allocation(worker_allocation, pc):
    n = worker_allocation.shape[1]
    popSize = worker_allocation.shape[0]
    worker_allocation1 = worker_allocation[:int(popSize/2)]
    worker_allocation2 = worker_allocation[int(popSize/2):]
    new_v1 = np.zeros_like(worker_allocation1)
    new_v2 = np.zeros_like(worker_allocation2)
    
    for i in range(int(popSize/2)):
        r = np.random.uniform()
        v1 = worker_allocation1[i]
        v2 = worker_allocation2[i]
        if r<=pc:
            p = np.random.randint(n)
            
            s1 = v1[p:]
            s2 = v2[p:]
            
            sorted_s1 = np.sort(s1)
            sorted_s2 = np.sort(s2)
            
            temp = np.vstack([sorted_s1, sorted_s2])
            temp1 = pd.DataFrame(data=temp[1, :], index=temp[0, :])
            temp2 = pd.DataFrame(data=temp[0, :], index=temp[1, :])
            
            new_v1[i] = np.hstack([v1[:p], temp2.loc[list(s2)].values.flatten()])
            new_v2[i] = np.hstack([v2[:p], temp1.loc[list(s1)].values.flatten()])
        else:
            new_v1[i] = v1.copy()
            new_v2[i] = v2.copy()
            
    new_v = np.vstack([new_v1, new_v2])
    
    for v in new_v:
        if 0 not in v or 1 not in v or 2 not in v or 3 not in v:
            print('AX_for_task error')
            print(v)
            print('='*20)
    return new_v

def swap_for_all(arr, pm):
    l = arr.shape[1]
    popSize = arr.shape[0]
    
    for i in range(popSize):
        r = np.random.uniform()
        new_arr = arr[i].copy()
        if r<=pm:
            point1, point2 = np.random.choice(l, size=2, replace=False)
            new_arr[point1], new_arr[point2] = arr[i, point2], arr[i, point1]
        arr[i] = new_arr
        
    return arr

def roulette_wheel_selection(old_P, P, old_F, F):
    popSize = int( ( len(P) + len(old_P) ) / 2 )
    
    P_set = np.vstack([old_P, P])
    F_set = np.hstack([old_F, F])
    F_sum = np.sum(F_set)
    
    pk = F_set/F_sum
    idx = np.argsort(pk)[::-1]
    sorted_pk = np.sort(pk)[::-1]
    
    qk = np.cumsum(sorted_pk)[::-1]
    qk = np.hstack([qk[1:], 0.0])
    
    p_idx = np.zeros(popSize).astype(int)
    r = np.random.uniform(size=[popSize])
    for i in range(len(r)):
        for j in range(len(qk)):
            if r[i]>qk[j]:
                p_idx[i] = idx[j]
                break
    
    new_P = P_set[p_idx]
    
    return new_P

def fitness(P, task_time_unit, task_cost_unit):
    popSize = len(P)
    z1 = np.zeros(popSize)
    z2 = np.zeros_like(z1)
    z3 = np.zeros_like(z1)
    
    for k in range(popSize):
        ######
        worker_allocation = P[k, 10:14]
        task_sequence = P[k, 14:24]
        breakpoint_vector = P[k, 24:]
        v2 = worker_allocation
        v3 = task_sequence
        v4 = breakpoint_vector.astype(int)
        m = len(v2)
        tS = np.zeros(m)
        dS = np.zeros_like(tS)
        dT = 0
        
        for i in range(m):
            worker = v2[i].astype(int) # 在崗位i值勤的工人
            Si = v3[v4[i]:v4[i+1]].astype(int) # 崗位i被分配的任務集
            tS[i] = np.sum(task_time_unit[Si, worker]) # 崗位i所需總工時
            dS[i] = np.sum(task_cost_unit[Si, worker]) # 崗位i所需總用料
            # dS[i] = len(Si)*100 - tS[i] # 偷雞的寫法，因為每一任務的材料和時間之總和為100
            
        u = tS / np.max(tS) # 每一崗位的運轉率
        cT = np.max(tS) # 最大工時
        v = ( ( ( u-u.mean() )**2 ).mean() )**.5 # 運轉率的rmse
        dT = dS.sum() # 總用料
        
        z1[k] = 1/cT
        z2[k] = 1/v
        z3[k] = 1/dT
        ######
    
    z1_max = z1.max()
    z1_min = z1.min()
    z2_max = z2.max()
    z2_min = z2.min()
    z3_max = z3.max()
    z3_min = z3.min()
    
    if z1_max!=z1_min and z2_max!=z2_min and z3_max!=z3_min:
        # 0 0 0
        v1 = z1_min/(z1_max-z1_min)
        v2 = z2_min/(z2_max-z2_min)
        v3 = z3_min/(z3_max-z3_min)
        v = v1 + v2 + v3
        w1 = v1/v
        w2 = v2/v
        w3 = v3/v
    elif z1_max!=z1_min and z2_max!=z2_min and z3_max==z3_min:
        # 0 0 1
        w1 = 0.45
        w2 = 0.45
        w3 = 0.1
    elif z1_max!=z1_min and z2_max==z2_min and z3_max!=z3_min:
        # 0 1 0
        w1 = 0.45
        w2 = 0.1
        w3 = 0.45
    elif z1_max!=z1_min and z2_max==z2_min and z3_max==z3_min:
        # 0 1 1
        w1 = 0.8
        w2 = 0.1
        w3 = 0.1
    elif z1_max==z1_min and z2_max!=z2_min and z3_max!=z3_min:
        # 1 0 0
        w1 = 0.1
        w2 = 0.45
        w3 = 0.45
    elif z1_max==z1_min and z2_max!=z2_min and z3_max==z3_min:
        # 1 0 1
        w1 = 0.1
        w2 = 0.8
        w3 = 0.1
    elif z1_max==z1_min and z2_max==z2_min and z3_max!=z3_min:
        # 1 1 0
        w1 = 0.1
        w2 = 0.1
        w3 = 0.8
    elif z1_max==z1_min and z2_max==z2_min and z3_max==z3_min:
        # 1 1 0
        w1 = 1/3
        w2 = 1/3
        w3 = 1/3
    else:
        print('fuck')
    
    r1 = np.random.uniform()
    r2 = np.random.uniform()
    r3 = np.random.uniform()
    F = ( w1*(z1_max-z1+r1)/(z1_max-z1_min+r1) ) + ( w2*(z2_max-z2+r2)/(z2_max-z2_min+r2) ) + ( w3*(z3_max-z3+r3)/(z3_max-z3_min+r3) )
    return F, 1/z1, 1/z2, 1/z3

#%% Initialization
predecessor = [[],
               [],
               [],
               [0],
               [1, 2],
               [3],
               [],
               [5],
               [],
               [6, 7, 8]]
task_time_unit = np.array([[17, 22, 19, 13],
                           [21, 22, 16, 20],
                           [12, 25, 27, 15],
                           [29, 21, 19, 16],
                           [31, 25, 26, 22],
                           [28, 18, 20, 21],
                           [42, 28, 23, 34],
                           [27, 33, 40, 25],
                           [19, 13, 17, 34],
                           [26, 27, 35, 26]])
task_cost_unit = 100 - task_time_unit
total_number_of_tasks = task_time_unit.shape[0]
total_number_of_stations = task_time_unit.shape[1]

pop_size = 20
maxGen = 500
pm = 0.2
pc = 0.6
pi = 0.0
times = 1
table = np.zeros(times)

#%% main

for run in range(times):
    # P[i] = [task_priorityv1(10) worker_allocation v2(4), task_sequence v3(10), breakpoint_vector v4(5)]
    st = time.time()
    P = []
    while len(P)<pop_size:
        v1 = task_priority(total_number_of_tasks)
        v2 = worker_allocation(total_number_of_stations)
        v3 = task_sequence(total_number_of_tasks, v1, predecessor)
        v4 = breakpoint_vector(v2, v3, task_time_unit)
        
        if len(v1)==10 and len(v2)==4 and len(v3)==10 and len(v4)==5:
            P.append(np.hstack([v1, v2, v3, v4]))
    P = np.array(P)
    
    F, cT, v, dT = fitness(P, task_time_unit, task_cost_unit)
    gbest_F = F.max()
    gbest_cT = cT[np.argmax(F)]
    gbest_v = v[np.argmax(F)]
    gbest_dT = dT[np.argmax(F)]
    gbest_v1 = P[np.argmax(F)][:10]
    gbest_v2 = P[np.argmax(F)][10:14].astype(int)
    gbest_v3 = P[np.argmax(F)][14:24].astype(int)
    gbest_v4 = P[np.argmax(F)][24:].astype(int)
    loss = [gbest_F]
    
    for t in range(maxGen):
        old_P = P.copy()
        old_F = F.copy()
        
        # 交配
        P[:, 10:14] = WMX_for_allocation(P[:, 10:14], pc) # v2
        P[:, :10] = AX_for_task(P[:, :10], pc) # v1
        
        # 突變
        P[:, 10:14] = swap_for_all(P[:, 10:14], pm) # v2
        P[:, :10] = swap_for_all(P[:, :10], pm) # v1
        
        for i in range(pop_size):
            aaa = task_sequence(total_number_of_tasks, P[i, 14:24], predecessor)
            P[i, 14:24] = aaa # v3
        
        for i in range(pop_size):
            aaa = breakpoint_vector(P[i, 10:14], P[i, 14:24], task_time_unit)
            if len(aaa)!=5:
                pass
            else:
                P[i, 24:] = aaa # v4
        
        # 適應值計算
        F, cT, v, dT = fitness(P, task_time_unit, task_cost_unit)
        if F.max()<gbest_F:
            gbest_F = F.max()
            gbest_cT = cT[np.argmax(F)]
            gbest_v = v[np.argmax(F)]
            gbest_dT = dT[np.argmax(F)]
            gbest_v1 = P[np.argmax(F)][:10]
            gbest_v2 = P[np.argmax(F)][10:14].astype(int)
            gbest_v3 = P[np.argmax(F)][14:24].astype(int)
            gbest_v4 = P[np.argmax(F)][24:].astype(int)
        loss.append(gbest_F)
        
        # 移民策略
        temp_F = np.hstack([F, old_F])
        temp_P = np.vstack([P, old_P])
        elite_idx = np.argsort(temp_F)[::-1][:pop_size]
        np.random.shuffle(elite_idx)
        F = temp_F[elite_idx]
        P = temp_P[elite_idx]
            
        # 選擇
        P = roulette_wheel_selection(old_P, P, old_F, F)
    
    print(str(run) + ', ' + str(gbest_F))
    print('工位1-工人' + str(gbest_v2[0]+1) + '-勤務' + str(gbest_v3[gbest_v4[0]:gbest_v4[1]]+1) + '-工時' + str(np.sum(task_time_unit[gbest_v3[gbest_v4[0]:gbest_v4[1]], gbest_v2[0]])) + '-成本' + str(np.sum((task_cost_unit[gbest_v3[gbest_v4[0]:gbest_v4[1]], gbest_v2[0]]))))
    print('工位2-工人' + str(gbest_v2[1]+1) + '-勤務' + str(gbest_v3[gbest_v4[1]:gbest_v4[2]]+1) + '-工時' + str(np.sum(task_time_unit[gbest_v3[gbest_v4[1]:gbest_v4[2]], gbest_v2[1]])) + '-成本' + str(np.sum((task_cost_unit[gbest_v3[gbest_v4[1]:gbest_v4[2]], gbest_v2[1]]))))
    print('工位3-工人' + str(gbest_v2[2]+1) + '-勤務' + str(gbest_v3[gbest_v4[2]:gbest_v4[3]]+1) + '-工時' + str(np.sum(task_time_unit[gbest_v3[gbest_v4[2]:gbest_v4[3]], gbest_v2[2]])) + '-成本' + str(np.sum((task_cost_unit[gbest_v3[gbest_v4[2]:gbest_v4[3]], gbest_v2[2]]))))
    print('工位4-工人' + str(gbest_v2[3]+1) + '-勤務' + str(gbest_v3[gbest_v4[3]:gbest_v4[4]]+1) + '-工時' + str(np.sum(task_time_unit[gbest_v3[gbest_v4[3]:gbest_v4[4]], gbest_v2[3]])) + '-成本' + str(np.sum((task_cost_unit[gbest_v3[gbest_v4[3]:gbest_v4[4]], gbest_v2[3]]))))
    print(str(np.round(time.time()-st, 3)) + 'sec')
    print('='*20)
    table[run] = gbest_F