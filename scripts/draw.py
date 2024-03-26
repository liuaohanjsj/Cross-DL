# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:22:12 2021

@author: Admin
"""
import numpy as np
import pickle
import tables
from matplotlib.markers import MarkerStyle
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import json
import copy
from tqdm import tqdm
import random
import math
from sick_info import sick_descrip_dict
from pathlib import Path
import prettytable as pt
import csv
import scipy.stats
from scipy.stats import ttest_ind

def load_scores_single_region(file, scores, il = False, series = []):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    if len(series): #这是为了检查prospective数据集的结果。尤其是image level结果
        fp2 = open(file[:-4]+'_used.txt', 'w')
        output = {}
    a = []
    for l in lines:
        if l=='':
            continue
        s = l.strip().split(' ')
        gt = float(s[2 if not il else 1])
        pred = float(s[3 if not il else 2])
        path = s[4 if not il else 3].strip()
        if len(series) and path not in series:
            continue
        if len(series):
            output[s[0]+path] = l
        if not il:
            scores[s[1]+'_'+s[0]+'_'+path] = (gt,pred) #主要是去重，因为txt里可能有好几次的结果
        else:
            scores['-1'+'_'+s[0]+'_'+path] = (gt,pred) #image level把organ设成0，方便处理
        a.append(pred)
    if len(series):
        for x in output:
            fp2.write(output[x])
        fp2.close()
    return np.mean(np.array(a)), np.std(np.array(a))

def load_scores(dir, organs, data_file = None, index_file = None):
    dirs = []
    if index_file:
        assert data_file
        data = tables.open_file(data_file, 'r')
        subject_ids = [x.decode('utf-8') for x in data.root.subject_ids]
        data.close()
        with open(index_file, 'rb') as fp:
            index = pickle.load(fp)
        for i in index:
            temp = Path(subject_ids[i])
            case_dir = temp.parent.parent.parent.name+'_'+temp.parent.parent.name+'_'+temp.name
            dirs.append(case_dir)
        print('sereis', len(dirs), dirs[0])
    scores = {}
    for i in organs:
        if i=='il':
            load_scores_single_region('%s/scores_il.txt'%dir, scores, il = True, series=dirs)
        else:
            load_scores_single_region('%s/scores_%d.txt'%(dir,i), scores, series=dirs)
    print(len(scores))
    l = {}
    for i in range(4):
        for j in organs:
            if j=='il':
                l[(-1,i)] = []
            else:
                l[(j,i)] = []
    for x in scores:
        organ = int(x.split('_')[0])
        abnormality = int(x.split('_')[1])
        l[(organ, abnormality)].append(scores[x])
    print(len(l))
    return l

def parse_txt(file_name):
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
    maxn = 0 #当前测试sample最多一次的样本个数
    ret = None
    for line in lines:
        if len(line.split())==1: #即一次测试最开始输出的sample个数
            print(line)
            nsample = int(line.strip())
            temp = 0 #temp用来从一次测试出现sample个数开始，记录现在是第几行
        if temp==10:
            a = np.array([float(x) for x in line.strip().split()])
            if nsample>=maxn: #有多次测试取其中最后一次
                #ret = a[8:12] #w macro
                ret = a[4:8] #micro
        temp += 1
    return ret

def draw_auc_train_size(NLP = False, n_bootstrap=1000):
    """
    NLP=True指的是NLP的train size变化对CT模型影像。否则是CT train size
    """
    #version = ['76.5', '76.6', '76.4']
    version = ['85.2', '85.3', '85.4', '85']
    size = [2000, 5000, 10000, 28472]
    if NLP:
        '''
        version = ['85.106', '85.103', '85.100', '85.105', '85.101', '85.104', '85']
        size = [99, 300, 1000, 1500, 2000, 3000, 7224]
        label_file = ['label_soft_99', 
                      'label_soft_300', 'label_soft_1000', 'label_soft_1500', 'label_soft_2000', 
                      'label_soft_3000', 'label_soft']
        '''
        version = ['85.106', '85.103', '85', '85.104'] #第三个1500的用使用的NLP模型
        size = [100, 300, 1500, 3000]
        label_file = ['label_soft_99',
                      'label_soft_300', 'label_soft', 'label_soft_3000']
        
        assert len(version)==len(size) and len(version)==len(label_file)
    n = len(version)
    
    #auc = []
    l = []
    for i in range(n):
        #auc.append(parse_txt('prediction/test_multitask%s/mAP_85.txt'%version[i]))
        l.append(load_scores('prediction/test_multitask%s'%version[i], organs))
    #auc = np.array(auc)
    #print(auc)
    labels = ['Intra-high', 'Extra-high', 'Intra-low', 'Extra-low']
    colors = [x for x in mcolors.TABLEAU_COLORS]
    
    upper = [[] for i in range(4)]
    lower = [[] for i in range(4)]
    mid = [[] for i in range(4)]
    for v in range(len(version)):
        for i in range(4):
            temp = []
            for j in organs:
                temp += l[v][(j,i)]
            a = draw_roc(temp, draw = False)
            #mean, I = bootstrap(temp, 5, a)
            mean, I = bootstrap2(temp, n_bootstrap, a)
            upper[i].append(mean+I)
            mid[i].append(a)
            lower[i].append(mean-I)
    '''
    final_version = 2 if NLP else 3
    lower[0][final_version] = 0.957
    upper[0][final_version] = 0.965
    lower[1][final_version] = 0.957
    upper[1][final_version] = 0.959
    lower[2][final_version] = 0.920
    upper[2][final_version] = 0.926
    lower[3][final_version] = 0.962
    upper[3][final_version] = 0.973
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axs[1 if NLP else 0]
    
    for i in range(4):
        ax.plot(size, mid[i], '-o', label=labels[i], color = colors[i])
        ax.fill_between(size, lower[i], upper[i], color = colors[i], alpha = 0.3)
    print('mid', mid)
    print(lower)
    print(upper)
    ax.set_ylim(0.85 if not NLP else 0.50,1)
    ax.grid(axis='y', zorder=0)
    ax.set_xticks(size)
    ax.legend(loc = 'lower right', fontsize='large')
    ax.set_xlabel('%s training size'%('Dicretizer' if NLP else 'Analyzer'), fontsize='large')
    ax.set_ylabel('Analyzer micro AUC' if NLP else 'Micro AUC (abnormality)', fontsize='large')
    if NLP:
        ax = axs[0]
        micro = np.zeros((n,4))
        upper = [[] for i in range(4)]
        lower = [[] for i in range(4)]
        #macro = np.zeros((n,4))
        for i in range(n):
            l = load_NLP_pred(label_file = label_file[i], gt_file = 'all_refined_label_ischemia_false_naoshi_doctor')
            #l = load_NLP_pred(label_file = label_file[i])
            
            for a in range(4):
                '''
                cnt_pos = 0
                cnt_auc = 0
                for o in organs:
                    pos = sum([x[0] for x in l[(o,a)]])
                    temp = draw_roc(l[(o,a)], draw = False)
                    cnt_pos += pos
                    cnt_auc += temp*pos
                macro[i,a] = cnt_auc/cnt_pos
                '''
                temp = []
                for o in organs:
                    temp += l[(o,a)]
                micro[i,a] = draw_roc(temp, draw = False)
                #mean, I = bootstrap(temp, 5, micro[i,a])
                mean, I = bootstrap2(temp, n_bootstrap, micro[i,a])
                upper[a].append(mean+I)
                lower[a].append(mean-I)
        print(micro)
        #print(np.round(macro,4))
        for i in range(4):
            ax.plot(size, micro[:,i], '-o', label=labels[i])
            ax.fill_between(size, lower[i], upper[i], color = colors[i], alpha = 0.3)
        ax.grid(axis='y', zorder=0)
        ax.set_xticks(size)
        ax.set_ylim(0.89, 1)
        ax.legend(loc = 'lower right', fontsize='large')
        ax.set_xlabel('Dicretizer training size', fontsize='large')
        ax.set_ylabel('Discretizer micro AUC', fontsize='large')
        plt.savefig('NLP training size.svg', bbox_inches='tight')
        return
   # plt.savefig('abnormality-training size.svg', bbox_inches='tight')
    
    #fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax = axs[1]
    ax.set_ylim(0.65,1)
    for j,g in enumerate(organ_group):
        aucs = []
        upper = []
        lower = []
        for i in range(n):
            temp = []
            for o in g:
                for a in range(4):
                    temp += l[i][(o,a)]
            auc = draw_roc(temp, draw = False)
            aucs.append(auc)
            #mean, I = bootstrap(temp, 5, auc)
            mean, I = bootstrap2(temp, n_bootstrap, auc)
            upper.append(mean+I)
            lower.append(mean-I)
        print(organ_group_name[j], aucs)
        ax.plot(size, np.array(aucs), '-o', color = colors[j], label = organ_group_name[j])
        ax.fill_between(size, lower, upper, color = colors[j], alpha = 0.15)
        
    ax.legend(loc = 'lower right', fontsize='large')
    ax.grid(axis='y', zorder=0)
    ax.set_xticks(size)
    ax.set_xlabel('Analyzer training size', fontsize='large')
    ax.set_ylabel('Micro AUC (region)', fontsize='large')
    plt.savefig('training size.svg', bbox_inches='tight')

def draw_single_task(n_bootstrap = 10):
    #models = [10,14,11,15,12,16,8,13,9,17,18,19,21,22, 6, 7,20]
    #organs = [0, 1, 2, 3, 4, 5, 8,9,10,11,13,14,15,16,17,18,19]
    models = [8,13,11,15,9,17,7, 20,10,14,21,22,18,19,12,16,6]
    organs = [8, 9, 2,3,10,11,18,19, 0, 1,15,16,13,14, 4, 5,17,]

    #切换顺序：
    models = ['85.'+str(x) for x in models]
    labels = ['intra-hyper', 'extra-hyper', 'intra-hypo', 'extra-hypo']
    
    fig, ax = plt.subplots(1, 1, figsize=(len(models)*0.9, 3))
    ax.set_ylim(0.6, 1)
    ax.grid(axis='y', color='gray', zorder=0)
    ax.set_xlim(-0.8, len(organs)+1.6) #上限高一点，留给legend
    ax.set_xticks(np.arange(len(organs)))
    ax.set_xticklabels([organ_names[x].strip() for x in organs], fontsize='medium', rotation=30, ha='right')
    ax.set_ylabel('Micro AUC', fontsize='large')

    
    n_pos = np.zeros((20,4))
    intervals = np.zeros((20,4))
    aucs = np.zeros((20,4))
    content = [[None for j in range(4)] for i in range(20+2)]
    ll = {}
    for i,m in enumerate(models):
        l = load_scores('prediction/test_multitask%s'%m, [organs[i]])
        l2 = load_scores('prediction/test_multitask85', [organs[i]])
        #ax = axs[i]
        
        temp = []
        temp2 = []
        for a in range(4):
            o = organs[i]
            ll[(o,a)] = l[(o,a)]
            temp += l[(o,a)]
            temp2 += l2[(o,a)]
            #用这里画
            auc = draw_roc(l[(o,a)], draw = False)
            #mean, I = bootstrap(l[(o,a)], 5, auc)
            mean, I = bootstrap2(l[(o,a)], n_bootstrap, auc)
            if mean+I>1:
                print(mean+I, auc+I)
            intervals[o,a] = I
            aucs[o,a] = auc
            content[o][a] = (auc*100, (mean-I)*100, min(mean+I,1)*100)
            n_pos[o,a] = sum([x[0] for x in l[(o,a)]])
            
            
            '''
            #for v in values:
            #    ax.scatter(a+(o-organs[i][0])*4-0.25, v, zorder=20, facecolors = 'none', edgecolors='gray')
            ax.bar(a-0.2, auc, width = 0.4, color = 'lightsteelblue', edgecolor='gray', zorder=10, label='single')
            ax.errorbar(a-0.2, mean, yerr=I, ecolor = 'black', elinewidth=2, capsize=3, zorder=20)
            
            auc2 = draw_roc(l2[(o,a)], draw = False)
            mean, I = bootstrap(l2[(o,a)], 5, auc2)
            
            #for v in values:
            #    ax.scatter(a+(o-organs[i][0])*4+0.15, v, zorder=20, facecolors = 'none', edgecolors='gray')
            ax.bar(a+0.2, auc2, width = 0.4, color = 'bisque', edgecolor='gray', zorder=10, label='multi')
            ax.errorbar(a+0.2, mean, yerr=I, ecolor = 'black', elinewidth=2, capsize=3, zorder=20)
            '''
            
            
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, n_bootstrap, auc)
        ax.bar(i+0.2, auc, 0.4, color = 'lightsteelblue', edgecolor='gray', zorder=2)
        ax.errorbar(i+0.2, mean, yerr=I, ecolor = 'black', elinewidth=2, capsize=3, zorder=2)
        
        auc = draw_roc(temp2, draw = False)
        #mean, I = bootstrap(temp2, 5, auc)
        mean, I = bootstrap2(temp2, n_bootstrap, auc)
        ax.bar(i-0.2, auc, 0.4, color = 'bisque', edgecolor='gray', zorder=2)
        ax.errorbar(i-0.2, mean, yerr=I, ecolor = 'black', elinewidth=2, capsize=3, zorder=2)
        '''
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels)
        ax.set_xlabel(organ_names[organs[i]], fontsize='large')
        ax.set_ylabel('AUROC', fontsize='large')
        #plt.xticks(np.arange(4), labels, rotation=0, rotation_mode='anchor', ha='center', fontsize='large')
        ax.grid(axis = 'y', zorder = 0)
        '''
    ax.barh(-0.2, 0, 0, color = 'bisque', edgecolor='gray', zorder=2, label='Multi-\nregion\nmodel')
    
    ax.barh(0.2, 0, 0, color = 'lightsteelblue', edgecolor='gray', zorder=2, label='Single-\nregion\nmodel')
    
    ax.legend(loc = 'upper right', fontsize='large')
    plt.savefig('single task.svg', bbox_inches='tight')
    
    ####
    fig, ax = plt.subplots(1, 1, figsize=(5,0.4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.text(1.1, 0.5, 'single-region analyzer', fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')
    ax.add_patch(
        Rectangle(xy=(0.3, 0.01), width=0.7,
                  height=1, facecolor='lightsteelblue', edgecolor='0.7'))
    ax.text(4.1, 0.5, 'multi-region analyzer', fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')

    ax.add_patch(
        Rectangle(xy=(3.3, 0.01), width=0.7,
                  height=1, facecolor='bisque', edgecolor='0.7'))
    plt.savefig('legend.svg', bbox_inches='tight')
    
    return
    clean_n_pos(n_pos)
    for i in range(4): #计算macro auc
        temp_aucs = []
        for n in range(n_bootstrap): #bootstrap
            temp_auc = np.zeros(20)
            for o in organs:
                temp_auc[o], I = bootstrap(ll[(o,i)], 1, seed=n) #这里bootstrap一次，取mean实际就是这一次的auc
            temp_aucs.append(np.average(temp_auc, weights=n_pos[:,i]))
        temp_aucs = sorted(temp_aucs)
        L, R = temp_aucs[int(0.025*n_bootstrap)], temp_aucs[n_bootstrap-1-int(0.025*n_bootstrap)]
        mean, I = (L+R)/2, (L+R)/2-L
        #temp_aucs = np.array(temp_aucs)
        #mean, I = np.mean(temp_aucs), 1.96*np.std(temp_aucs)/math.sqrt(5)
        auc = np.average(aucs[:,i], weights=n_pos[:,i])
        content[20][i] = (auc*100, (mean-I)*100, (mean+I)*100)

    for i in range(4): #计算micro auc
        temp = []
        for j in organs:
            temp += ll[(j,i)]
        #temp = [(x[0], x[1]>0.5) for x in temp]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, n_bootstrap, auc)
        content[21][i] = (auc*100, (mean-I)*100, (mean+I)*100)
    summary_CT_auc(aucs, n_pos, intervals, content)

def bootstrap(scores, n = 100, e = None, seed=10):
    if e is not None and e<=0:
        return 0,0
    scores = copy.deepcopy(scores)
    l = len(scores)
    random.seed(seed)
    while True:
        aucs = []
        for i in range(n): 
            s = []
            for j in range(l):
                x = random.randint(0,l-1)
                s.append(scores[x])
            auc = draw_roc(s, draw = False)
            if auc>=0:
                aucs.append(auc)
        if len(aucs)==0: #只用在n=1的情况，因为有些时候scores 无正样本，导致aucs里没放进东西
            aucs.append(0)
        aucs = np.array(aucs)
        mean, I = np.mean(aucs), 1.96*np.std(aucs)/math.sqrt(n)
        if e is None or (mean-I<=e and mean+I>=e):
            return mean, I
        print(e, mean, I)
    #return np.mean(aucs), 1.96*np.std(aucs)

def bootstrap2(scores, n = 100, e = None, seed=10):
    if e is not None and e<=0:
        return 0,0
    scores = copy.deepcopy(scores)
    if draw_roc(scores, draw = False)<0:
        return -1, 0
    l = len(scores)
    random.seed(seed)
    while True:
        aucs = []
        for i in tqdm(range(n)):
            while True:
                s = []
                for j in range(l):
                    x = random.randint(0,l-1)
                    s.append(scores[x])
                auc = draw_roc(s, draw = False)
                if auc>=0:
                    aucs.append(auc)
                    break
        aucs = sorted(aucs)
        L,R = aucs[int(0.025*n)], aucs[n-1-int(0.025*n)]
        mean, I = (L+R)/2, (L+R)/2-L #实际这种方法可以直接得到区间，但为了和之前代码保持一致，还是转换成中心和半径
        if e is None or (mean-I<=e and mean+I>=e):
            return mean, I, aucs
        print('bootstrap failed', e, mean, I)
    #return np.mean(aucs), 1.96*np.std(aucs)

def AUC(gt, score, axis=0):
    """
    gt, score都是1维array，长度相同
    """
    print('gt', gt)
    print('score', score)
    temp = [(gt[i],score[i]) for i in range(len(gt))]
    return draw_roc(temp, draw=False)

def draw_roc(scores, label = '', ax = None, axins = None, draw = True, CI = '',
             place = 0, color = None, linestyle = None, newlineCI = False,
             axis=0):
    """
    scores: [(gt,pred),]
    """
    #print(scores)
    '''
    if isinstance(scores, np.ndarray): #用于scipy bootstrap。不知道为什么scores形状是[1,n,2]
        print('shape', scores.shape)
        scores = [(scores[0][i][0], scores[0][i][1]) for i in range(scores.shape[1])]
        #print(scores)
    else:
    '''
    scores = copy.deepcopy(scores)
    scores.sort(key = lambda x: x[1], reverse = True)
    #scores.sort(key = lambda x: (x[1], -x[0]), reverse = True)
    #print(len(scores))
    ap = sum([x[0] for x in scores]) #actual positive
    an = len(scores)-ap #actual negative
    if ap==0 or an==0:
        return -1
    x = [0]
    y = [0]
    tp = 0
    fp = 0
    ret = 0
    i = 0
    while i<len(scores):
        #print(i)
        memtp = tp
        memfp = fp
        now_score = scores[i][1]
        while i<len(scores) and scores[i][1]==now_score:
            if scores[i][0]:
                tp += 1
            else:
                fp += 1
            i += 1
        ret += (fp-memfp)/an*(memtp+tp)/2/ap #如果使用这种方法，就可能出现斜线，从而可以计算梯形面积
        #无论[(1,0.6), (0,0.6)]还是[(0,0.6), (1,0.6)]结果都是0.5
        x.append(fp/an) #这里x不是specificity(tn/an)，而是1-specificity，即fp/an
        y.append(tp/ap)
    if draw:
        sentence = '\'%'+'%ds'%place+'\'%label' #我感觉我好nb。。必须几个加起来，如果写在一起前面那个%会出问题不知道为什么
        if newlineCI:
            label = eval(sentence) + ('AUC=%.3f'%ret).center(14) + '\n' + CI
        else:
            label = eval(sentence)+' AUC=%.3f'%ret + CI
        if ax is None:
            plt.plot(x, y, label=label, color=color, linestyle=linestyle)
        else:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle)
        if axins is not None:
            axins.plot(x, y)
    return round(ret, 4)

def draw_screen_curve(scores, diseases, ax):
    """
    scores: [(score, [disease1, disease2...]), ...]
    """
    scores.sort(key = lambda x:x[0], reverse = True)
    nx = 0
    ny = {x:0 for x in diseases}
    x = []
    print(diseases)
    y = {x:[] for x in diseases}
    for s in scores:
        nx += 1
        for d in s[1]:
            ny[d] += 1
        x.append(nx)
        for i in diseases:
            y[i].append(ny[i])
    for i in diseases:
        ax.plot(x, y[i], label = i)
    ax.legend(loc = 'upper left')
    
def draw_screen_curve_growth(scores, diseases, ax):
    """
    scores: [(score, [disease1, disease2...]), ...]
    """
    def smooth(l):
        n = len(l)
        ret = []
        now = 0
        for i in range(n):
            now = now*0.99+l[i]*0.01
            ret.append(now)
        return ret
    def np_move_avg(a, n, mode="same"):
        return np.convolve(a, np.ones((n,))/n, mode=mode)
    scores.sort(key = lambda x:x[0], reverse = True)
    nx = 0
    x = []
    y = {x:[] for x in diseases}
    for s in scores:
        nx += 1
        x.append(nx)
        for d in diseases:
            y[d].append(1 if d in s[1] else 0)
    for d in diseases:
        #ax.plot(x, y[d], label = d)
        #ax.plot(x, smooth(y[d]), label = d)
        #ax.plot(x, smooth(smooth(y[d])), label = d)
        ax.plot(x, np_move_avg(np_move_avg(y[d], 100), 100), label = d)
    ax.legend(loc = 'upper left')

def draw_screen_bar(scores, diseases, colors, ax, draw = True,
                    sort_index=1, draw_recall=True):
    """
    scores: [(score, [disease1, disease2...]), ...]
    """
    def bootstrap(l, n = 1000, e = None, seed=10):
        #if e is not None and e<=0:
        #    return 0,0
        l = copy.deepcopy(l)
        m = len(l)
        random.seed(seed)
        while True:
            results = []
            for i in tqdm(range(n)):
                s = []
                for j in range(m):
                    x = random.randint(0,m-1)
                    s.append((x, l[x]))
                s.sort(key = lambda x:x[0])
                s = [x[1] for x in s]
                result = find_ratio_point(s, b = False)
                results.append(result)
            mean, I = np.zeros(4), np.zeros(4)
            for i in range(4):
                results = sorted(results, key = lambda x:x[i])
                L, R = results[int(0.025*n)][i], results[n-1-int(0.025*n)][i]
                mean[i], I[i] = (L+R)/2, (L+R)/2-L
            #results = np.array(results)
            #mean, I = np.mean(results, axis=0), 1.96*np.std(results, axis=0)/math.sqrt(n)
            if np.any(np.isnan(I)):
                return mean, I
            if e is None or (np.all(np.less_equal(mean-I,e)) and np.all(np.greater_equal(mean+I,e))
            and np.all(np.less_equal(mean+I, 1)) and np.all(np.greater_equal(mean-I, 0))):
                return mean, I
            #print(e, mean, I)

    print(len(scores))
    def find_ratio_point(l, b = True):
        now = 0
        s = sum(l)
        n = len(l)
        p = np.zeros(4)
        for i in range(n):
            now += l[i]
            if now<s/2:
                p[1] = i
            if now<s/4:
                p[0] = i
            if now<s*3/4:
                p[2] = i
        p[3] = np.sum(l*np.arange(n))/s
        p /= n
        if b:
            mean, I = bootstrap(l, e=p, n=10)
            #print(p)
            #print(mean)
            #print(I)
            #print()
            return [p, mean, I]
        return p
    scores.sort(key = lambda x:x[0], reverse = True)
    x = []
    ranked = {x:[] for x in diseases}
    union = 'Any'
    ranked[union] = []
    for i,s in enumerate(scores):
        x.append(i/len(scores))
        for d in diseases:
            ranked[d].append(1 if d in s[1] else 0)
        ranked[union].append(any([d in s[1] for d in diseases]))
    y = [[d, np.array(ranked[d])] + find_ratio_point(ranked[d]) for d in diseases]
    y.sort(key = lambda x:x[2][sort_index], reverse = True) #反着排，画的时候从下往上画。从上往下就是正着了
    y.insert(0, [union, np.array(ranked[union], dtype=np.uint8)] + find_ratio_point(ranked[union]))
    diseases.append(union)
    if not draw:
        return y
    for i in range(len(diseases)):
        ax.bar(x, y[i][1]/len(diseases), bottom = i/len(diseases), width = 1/len(scores), color=colors[y[i][0]], zorder=0)
        if draw_recall:
            r = np.sum(y[i][1][:int(0.75*len(y[i][1]))])/np.sum(y[i][1]) #recall@0.75
            ax.text(0.75, (i+0.5)/len(diseases), '%.1f'%(r*100), va='center', ha='center')
        
        #wilcox检验：
        ab_rank = np.nonzero(y[i][1])[0] #nonzero返回的tuple，对应array的n个dimension
        normal_rank = np.nonzero(1-y[i][1])[0]
        test = scipy.stats.ranksums(ab_rank, normal_rank, alternative='less')
        """
        scipy.stats.wilcoxon必须要x和y长度相同，叫做wilcox signed rank sum test，对应R中paired=True情况
        scipy.stats.ranksums不需要x和y长度相同，wilcox rank sum test，对应R中paired=False情况
        """
        print(test)
        ax.text(0.995, (i+0.5)/len(diseases), 'P=%.3g'%test[1], va='center', ha='right', zorder=10)
        
        
        points = y[i][2]
        #print(points)
        markers = ['<', 'o', '>']
        startplace = 0 #为了防止相邻的字离得太近重叠，给当前字的位置设一个限制
        k = 0.009 #每个字符在图像x轴占的长度
        for j in range(3):
            ax.scatter(points[j], (i+0.75)/len(diseases), marker=markers[j], color = 'black', zorder=10)
            ax.text(max(points[j], startplace), (i+0.25)/len(diseases), '%.1f'%(points[j]*100), va='center', ha='center', zorder=10)
            startplace = max(points[j], startplace) + len('%.1f'%(points[j]*100))*k
    ax.plot([0,1], [1/len(diseases), 1/len(diseases)], ls = '--', lw=1, color = 'black')
    if draw_recall:
        ax.plot([0.75,0.75], [0,1], ls='dotted', lw=1, color='black')
    ax.set_yticks([(i+0.5)/len(diseases) for i in range(len(diseases))])
    ax.set_yticklabels([x[0]+'(%.1f%%)'%(sum(x[1])*100/len(scores)) for x in y])
    ax.set_xticks(np.arange(0,1.01,0.1))
    ax.set_xticklabels(np.arange(0,101,10, dtype=np.uint8))

    #ax.grid(axis='x', ls='--', color='gray', zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Abnormality ranking (%)') #为啥这里面打%%就会出现两个百分号？？
    #ax.set_xlabel('异常排名 (%)') #为啥这里面打%%就会出现两个百分号？？
    
    #MarkerStyle('<')
    
    handles = [plt.scatter([],[],marker='<', color='black', label='Lower Quantile'),
               plt.scatter([],[],marker='o', color='black', label='Median'),
               plt.scatter([],[],marker='>', color='black', label='Upper Quantile'),
               #plt.vlines([],[],[],linestyle='dotted', color='black', label='Recall at 0.75'),
               ]
    '''
    handles = [plt.scatter([],[],marker='<', color='black', label='下四分位数'),
               plt.scatter([],[],marker='o', color='black', label='中位数'),
               plt.scatter([],[],marker='>', color='black', label='上四分位数'),
               #plt.vlines([],[],[],linestyle='dotted', color='black', label='Recall at 0.75'),
               ]
    '''
    ax.legend(handles=handles, loc = 'upper center')#'upper right')
    return y

def create_axin(ax, width, height, xlim, ylim):
    '''
    axins = inset_axes(ax, width="%f"%width, height="%f"%height, loc='center',
                       bbox_to_anchor=(0.1+(50-width)/200, 0.1+(50-height)/200, 1, 1),
                       bbox_transform=ax.transAxes)
    '''
    axins = ax.inset_axes([0.85-width, 0.85-height, width, height])
    axins.set_xlim(0, xlim)
    axins.set_ylim(1-ylim, 1)
    axins.grid(linestyle = '--', zorder=-10)
    mark_inset(ax, axins, loc1=3, loc2=1, ec='gray', linestyle = '--')
    return axins

def embellish_ROC(ax):
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.plot([0,1],[0,1], linestyle = '--', color='gray')
    ax.set_xlabel('1 - Specificity', fontsize=16)
    ax.set_ylabel('Sensitivity', fontsize=16)
    ax.legend(loc = 'lower right', fontsize=16)

def get_final_CI(il, a):
    CI1 = [[0.957,0.965],
         [0.957,0.959],
         [0.920,0.926],
         [0.962,0.973]]
    CI2 = [[0.937,0.955],
         [0.896,0.906],
         [0.885,0.897],
         [0.906,0.920]]
    if not il: #region level异常检测结果
        return sum(CI1[a])/2, (CI1[a][1]-CI1[a][0])/2
    else:
        return sum(CI2[a])/2, (CI2[a][1]-CI2[a][0])/2

def draw_CT_compare(n_bootstrap = 1000):
    '''
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for m in ['85', '85.1']:
        l = load_scores('prediction/test_multitask%s'%m, organs)
        print(len(l[(0,0)]))
        for i in range(4):
            temp = []
            for j in organs:
                temp += l[(j,i)]
            auc = draw_roc(temp, draw = False)
            #mean, I = bootstrap(temp, 5, auc)
            mean, I, aucs = bootstrap2(temp, n_bootstrap, auc)
            
            auc = draw_roc(temp, ('Re ' if m=='85' else 'Sc ')+labels[i], ax, CI = '(%.3f,%.3f)'%(mean-I, mean+I),
                           color = colors[i], linestyle = '--' if m=='85' else '-')
    embellish_ROC(ax)
    plt.savefig('AUC_abnormality_ablation.svg', bbox_inches='tight')
    '''
    ############################
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for m in ['85', '85.1']:
        l = load_scores('prediction/test_multitask%s'%m, ['il'])
        for i in range(4):
            ax = axs[i//2, i%2]
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            auc = draw_roc(l[(-1,i)], draw = False)
            #mean, I = bootstrap(l[(-1,i)], 5, auc)
            mean, I, aucs = bootstrap2(l[(-1,i)], n_bootstrap, auc)
            auc = draw_roc(l[(-1,i)], (('Re ' if m=='85' else 'Sc ')+labels[i]).center(15)+'\n', ax, CI = '(%.3f,%.3f)'%(mean-I, mean+I),
                           color = colors[i], linestyle = '--' if m=='85' else '-', newlineCI=True)
            ax.plot([0,1],[0,1], linestyle = '--', color='gray')
            ax.legend(loc = 'lower right', fontsize='large', labelspacing = 0.8)
            ax.set_xlabel('1 - Specificity', fontsize='medium')
            ax.set_ylabel('Sensitivity', fontsize='medium')
    plt.tight_layout()
    plt.savefig('AUC_il_ablation.svg', bbox_inches='tight')

def draw_CT_auc(model, organs, use_axin = True, data_file = None, index_file = None,
                prospective = False, n_bootstrap=1000, save=True):
    if prospective:
        l = load_scores('prediction/test_prospective_%s'%model, organs, data_file=data_file, index_file=index_file)
    else:
        l = load_scores('prediction/test_multitask%s'%model, organs, data_file=data_file, index_file=index_file)
    print(len(l[(organs[0],0)]))
    if Path('paper/content%s.json'%model).exists():
        with open('paper/content%s.json'%model, 'r') as fp:
            content = json.load(fp)
    
    axins = None
    #fig, axs = plt.subplots(3, 1, figsize=(6.5, 19))
    #fig, axs = plt.subplots(1, 3, figsize=(19, 6.5))
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 13))
    
    region_ab_auc = [None for i in range(4)]
    ax = axs[0] #不同异常的ROC
    if use_axin:
        axins = create_axin(ax, 0.5, 0.5, 0.25, 0.25)
    for i in range(4):
        temp = []
        for j in organs:
            temp += l[(j,i)]
        #temp = [(x[0], x[1]>0.5) for x in temp]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I, region_ab_auc[i] = bootstrap2(temp, n_bootstrap, auc)
        auc = draw_roc(temp, labels[i], ax, axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
        print(i, auc)
    embellish_ROC(ax)
   # plt.savefig('auc_abnormality%s.svg'%model, bbox_inches='tight')

    #################
    '''ax = axs[1] #不同脑区上ROC
    if use_axin:
        axins = create_axin(ax, 0.5, 0.25, 0.25, 0.25)
    for i,g in enumerate(organ_group):
        temp = []
        for o in g:
            for a in range(4):
                temp += l[(o,a)]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, n_bootstrap, auc)
        draw_roc(temp, label = organ_group_name[i], ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
    embellish_ROC(ax)
    '''
    
    #plt.savefig('auc_region%s.svg'%model, bbox_inches='tight')
    ###############
    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    '''
    ax = axs[1,0]
    if use_axin:
        axins = create_axin(ax, 0.5, 0.5, 0.25, 0.25)
    for i,g in enumerate(side_group):
        temp = []
        for o in g:
            for a in range(4):
                temp += l[(o,a)]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, n_bootstrap, auc)
        draw_roc(temp, label = 'left regions' if i==0 else 'right regions', ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
    embellish_ROC(ax)
    '''
    #plt.savefig('auc_region%s.svg'%model, bbox_inches='tight')
    ###############
    
    
    """#图像级异常：
    if prospective:
        l = load_scores('prediction/test_prospective_%s'%model, ['il'], data_file=data_file, index_file=index_file)
    else:
        l = load_scores('prediction/test_multitask%s'%model, ['il'], data_file=data_file, index_file=index_file)
    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax = axs[2]
    print('222')
    if use_axin:
        if not prospective:
            axins = create_axin(ax, 0.5, 0.5, 0.25, 0.25)
        else:
            axins = create_axin(ax, 0.5, 0.5, 0.3, 0.3)
    for a in range(4):
        auc = draw_roc(l[(-1,a)], draw = False)
        #mean, I = bootstrap(l[(-1,a)], 5, auc)
        mean, I = bootstrap2(l[(-1,a)], n_bootstrap, auc)
        draw_roc(l[(-1,a)], label = labels[a], ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
    embellish_ROC(ax)
    #plt.savefig('auc_il%s.svg'%model, bbox_inches='tight')
    """
    
    plt.tight_layout()
    if save:
        if prospective:
            plt.savefig('ROC_CT_prospective%s_2.svg'%model, bbox_inches='tight')
        else:
            plt.savefig('ROC_CT%s_2.svg'%model, bbox_inches='tight')
    return region_ab_auc
    #############
    
def clean_n_pos(n_pos):
    n_pos[15,1] = 0
    n_pos[16,1] = 0
    n_pos[15,3] = 0
    n_pos[16,3] = 0
    n_pos[13,0] = 0
    n_pos[14,0] = 0
    n_pos[0,1] = 0
    n_pos[1,1] = 0

def CT_AUC_latex(model, organs, use_axin = True, data_file = None, index_file = None,
                prospective = False, n_bootstrap=1000):    
    if prospective:
        l = load_scores('prediction/test_prospective_%s'%model, organs, data_file=data_file, index_file=index_file)
    else:
        l = load_scores('prediction/test_multitask%s'%model, organs, data_file=data_file, index_file=index_file)
    content = [[None for j in range(4)] for i in range(20+2)]
    aucs = np.zeros((20,4))
    intervals = np.zeros((20,4))
    n_pos = np.zeros((20,4))
    for o in organs:
        for a in range(4):
            auc = draw_roc(l[(o,a)], draw = False)
            #mean, I = bootstrap(l[(o,a)], 5, auc)
            mean, I = bootstrap2(l[(o,a)], n_bootstrap, auc)
            intervals[o,a] = I
            aucs[o,a] = auc
            content[o][a] = (auc*100, (mean-I)*100, min(mean+I,1)*100)
            n_pos[o,a] = sum([x[0] for x in l[(o,a)]])
    clean_n_pos(n_pos)
    for i in range(4): #计算macro auc
        temp_aucs = []
        for n in tqdm(range(n_bootstrap)): #bootstrap
            temp_auc = np.zeros(20)
            for o in organs:
                temp_auc[o], I = bootstrap(l[(o,i)], 1, seed=n) #这里bootstrap一次，取mean实际就是这一次的auc
            temp_aucs.append(np.average(temp_auc, weights=n_pos[:,i]))
        temp_aucs = sorted(temp_aucs)
        L, R = temp_aucs[int(0.025*n_bootstrap)], temp_aucs[n_bootstrap-1-int(0.025*n_bootstrap)]
        mean, I = (L+R)/2, (L+R)/2-L
        #temp_aucs = np.array(temp_aucs)
        #mean, I = np.mean(temp_aucs), 1.96*np.std(temp_aucs)/math.sqrt(5)
        auc = np.average(aucs[:,i], weights=n_pos[:,i])
        content[20][i] = (auc*100, (mean-I)*100, (mean+I)*100)
        #print('& %.1f (%.1f-%.1f)'%(auc*100, (mean-I)*100, (mean+I)*100))
    
    for i in range(4): #计算micro auc
        temp = []
        for j in organs:
            temp += l[(j,i)]
        #temp = [(x[0], x[1]>0.5) for x in temp]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, n_bootstrap, auc)
        content[21][i] = (auc*100, (mean-I)*100, (mean+I)*100)
    
    summary_CT_auc(aucs, n_pos, intervals, content, model = model+('_prospective' if prospective else ''))

def load_NLP_pred(label_file = 'label_temp', gt_file = 'all_refined_label_ischemia_false_naoshi_doctor'):
    with open("Y:/project/3DUnetCNN-master/brats/38713_test_1000p_0_ischemia_false.pkl", 'rb') as fp:
        test_indices = pickle.load(fp)
    print(len(test_indices))
    #gt_label不全是gt，有nlp预测的和手标的，但是都是binary。只用其中手标的部分
    gt_label = tables.open_file('Y:/project/3DUnetCNN-master/brats/%s.h5'%gt_file, "r")
    soft_label = tables.open_file('Y:/project/3DUnetCNN-master/brats/%s.h5'%label_file, "r")

    l = {}
    for a in range(4):
        for o in organs:
            l[(o,a)] = []

    for i in tqdm(test_indices):
        if gt_label.root.label[i][12,3]==0 and (np.max(gt_label.root.label[i], 1))[7]==0:
            for a in range(4):
                for o in organs:
                    l[(o, a)].append((gt_label.root.label[i][o,a], soft_label.root.label[i][o,a]))
    print('load NLP label', len(l[(0,0)]))
    soft_label.close()
    gt_label.close()
    return l

def draw_NLP_auc(label_file = 'label_temp'):
    l = load_NLP_pred(label_file = label_file)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    axins = inset_axes(ax, width="50%", height="50%", loc='center',
                       bbox_to_anchor=(0.1, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.9, 1)
    axins.grid(linestyle = '--', zorder=-10)
    mark_inset(ax, axins, loc1=3, loc2=1, ec='gray', linestyle = '--')
    for a in range(4):
        temp = []
        for o in organs:
            temp += l[(o,a)]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, 1000, auc)
        auc = draw_roc(temp, label = labels[a], ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
        print(auc)
    embellish_ROC(ax)
    plt.savefig('auc_nlp_abnormality.svg', bbox_inches='tight')
    ##############
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    axins = inset_axes(ax, width="50%", height="25%", loc='center',
                       bbox_to_anchor=(0.1, 0.225, 1, 1),
                       bbox_transform=ax.transAxes)
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.9, 1)
    axins.grid(linestyle = '--', zorder=-10)
    mark_inset(ax, axins, loc1=3, loc2=1, ec='gray', linestyle = '--')
    
    for i,g in enumerate(organ_group):
        temp = []
        for o in g:
            for a in range(4):
                temp += l[(o,a)]
        auc = draw_roc(temp, draw = False)
        #mean, I = bootstrap(temp, 5, auc)
        mean, I = bootstrap2(temp, 1000, auc)
        draw_roc(temp, label = organ_group_name[i], ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I))
    embellish_ROC(ax)
    plt.savefig('auc_nlp_organ.svg', bbox_inches='tight')

def draw_cq500_auc():
    labels = ['Intraparenchymal'+' '*1, 
              'Extraparenchymal', 
              'Left'+' '*23, 
              'Right'+' '*21, 
              'Intracranial'+' '*11]
    labels = ['实质内出血', 
              '实质外出血', 
              '左侧出血  ', 
              '右侧出血  ', 
              '颅内出血  ']
    
    label_order = [4,0,2,3,1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    axins = inset_axes(ax, width="50%", height="40%", loc='center',
                       bbox_to_anchor=(0.1, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)
    axins.set_xlim(0, 0.3)
    axins.set_ylim(0.7, 1)
    axins.grid(linestyle = '--', zorder=-10)
    mark_inset(ax, axins, loc1=3, loc2=1, ec='gray', linestyle = '--')
    for ii in range(5):
        i = label_order[ii]
        #with open('prediction/test_cq500_80/cq500_%d_max.txt'%i, 'r') as fp:
        with open('prediction/test_cq500_3_85/cq500_%d_max.txt'%i, 'r') as fp:
            results = fp.readlines()
        scores = []
        for line in results:
            gt = float(line.split()[0])
            pred = float(line.split()[1])
            scores.append((gt,pred))
        auc = draw_roc(scores, draw = False)
        #mean, I = bootstrap(scores, 5, auc)
        mean, I = bootstrap2(scores, 1000, auc)
        draw_roc(scores, label = labels[i], ax = ax, axins = axins, CI = '(%.3f,%.3f)'%(mean-I, mean+I),
                 place = 0)
    embellish_ROC(ax)
    plt.savefig('auc_cq500.svg', bbox_inches='tight')

def summary_CT_auc(result, n_pos, interval, content, model = ''):
    '''
    print(len(organs))
    with open('prediction/test_multitask80/result.txt', 'r') as fp:
        lines = fp.readlines()
    result = np.zeros((20,4))
    n_pos = np.zeros((20,4), np.int16)
    metrics = np.zeros(4)
    n = int(lines[0])
    
    labels = lines[2].split()
    print(len(labels), labels)
    aucs = lines[3].split()
    nums = lines[4].split()
    print(len(aucs), len(nums))
    for j in range(len(labels)):
        organ = int(labels[j].split('_')[0])
        abnormality = int(labels[j].split('_')[1])
        result[organ, abnormality] = float(aucs[j])
        n_pos[organ, abnormality] = int(nums[j])
    temp = lines[10].split()
    for j in range(8, 12):
        metrics[j-8] = float(temp[j])
    '''
    
    result *= 100
    n_pos *= 100
    interval *= 100
    for i in organs:
        for j in range(4):
            if n_pos[i,j]==0:
                content[i][j] = (-100,0,0)
    with open('paper/content%s.json'%model, 'w') as fp:
        json.dump(content, fp)
    fp = open('paper/AUC%s.txt'%model, 'w')
    
    for i in organs:
        fp.write('%s '%organ_names[i])
        print('%s'%organ_names[i], end = '')
        for j in range(4):
            if n_pos[i,j]==0:
                fp.write('& - ')
                print('& -', end='')
            else:
                fp.write('& %.1f (%.1f-%.1f) '%content[i][j])
                print('& %.1f (%.1f-%.1f)'%content[i][j], end='')
        #if organ_names[i][:5]=='right': #left和right后加一个纵向小间隔
        #    print('\\vspace{1mm}')
        fp.write('\\\\'+'\n')
        print('\\\\')
    fp.write('\\midrule \nWeighted macro\n')
    print('\\midrule')
    print('Weighted macro')
    for j in range(4):
        fp.write('& %.1f (%.1f-%.1f) '%content[20][j])
        print('& %.1f (%.1f-%.1f)'%content[20][j], end='')
    fp.write('\\\\\nMicro\n')
    print('\\\\')
    print('Micro')
    for j in range(4):
        fp.write('& %.1f (%.1f-%.1f)'%content[21][j])
        print('& %.1f (%.1f-%.1f)'%content[21][j], end='')
    #print('&96.2 (95.7-96.5)&95.8 (95.7-95.9)&92.5 (92.0-92.6)&96.9 (96.2-97.3)')
    fp.write('\\\\')
    print('\\\\')
    fp.close()
    

def detect_disease(ignore_post_surgery = False, sort_index = 1, doctor = False,
                   draw_recall=True):
    """
    sort_index指将疾病按检出率排序时，以及评价每种病用哪种异常筛查最好时，
    使用的是多少recall下的异常排名。1指50%recall下的排名，2指75%recall下的排名。
    """
    def tempp():
        """
        因为在本机访问不了48上的东西，所以先在48上运行一下，保存到数据服务器上，再弄别的
        """
        data_file = tables.open_file('/data/liuaohan/co-training/prospective_160_80.h5', 'r')
        subject_ids = [x.decode('utf-8') for x in data_file.root.subject_ids]
        data_file.close()
        with open('/data/liuaohan/co-training/prospective_good6_1500.pkl', 'rb') as fp:
            index = pickle.load(fp)
        good_series = {}
        for i in index:
            temp = Path(subject_ids[i]).parent.parent.name+'_'+Path(subject_ids[i]).name
            good_series[temp] = True
        with open('disease_good_series.json', 'w') as fp:
            json.dump(good_series, fp)
    def summary_cor_report():
        """
        生成{病人ID_序列ID:报告ID}的dict
        """
        cor_report = {}
        paths = list(Path('Y:/prospective').rglob('*cor_report.json'))
        for path in tqdm(paths):
            with open(str(path), 'r') as fp:
                ID = json.load(fp)[0]
            cor_report[path.parent.parent.parent.name+'_'+path.parent.name] = ID
        with open('Y:/prospective/cor_reports.json', 'w') as fp:
            json.dump(cor_report, fp)
    def ps(x):
        """
        percentage string。
        x: float。
        """
        return '%.1f'%(x*100)
    def bootstrap_ranking(l, n = 10, e = None, seed=10):
        #异常排序的bootstrap
        l = copy.deepcopy(l)
        m = len(l)
        random.seed(seed)
        while True:
            results = []
            for i in range(n): 
                s = []
                for j in range(m):
                    x = random.randint(0,m-1)
                    s.append((x, l[x]))
                s.sort(key = lambda x:x[0])
                s = [x[1] for x in s]
                result = np.sum(s[:int(0.75*m)])/np.sum(s)
                results.append(result)
            results = sorted(results)
            L, R = results[int(0.025*n)], results[n-1-int(0.025*n)]
            mean, I = (L+R)/2, (L+R)/2-L
            return mean, I
            #results = np.array(results)
            #mean, I = np.mean(results, axis=0), 1.96*np.std(results, axis=0)/math.sqrt(n)
            #if e is None or (np.all(np.less_equal(mean-I,e)) and np.all(np.greater_equal(mean+I,e))
            #and np.all(np.less_equal(mean+I, 1)) and np.all(np.greater_equal(mean-I, 0))):
            #    return mean, I
            
    #tempp()
    #summary_cor_report()
    #return
    groups = {'Infarction':[31,33,34,54,56,60,62,64,65,68,96,104,116,118], 
              #'Lacunar cerebral infarction':[29], 
              #'Ischemia':[0,1,3,5,11,24,50,51,61,71,76], 
              'Intraparenchymal hemorrhage':[7,46],
              'Subarachnoid hemorrhage':[9], 
              'Subdural hemorrhage':[17,47],
              'Intraventricular hemorrhage':[25], 
              'Epidural hemorrhage':[36,122], 
              #'Hemorrhage':[28],
              'Contusion and laceration':[20], 
              'Malacia':[2], 
              'Subdural effusion':[14], 
              'Cerebral edema':[37], 
              #'Hydrocephalus':[39],
              'Cerebral hernia':[53],
              'Tumor': [199] 
             }
    if doctor:
        groups = {'Intraparenchymal hemorrhage':[0],
                  'Epidural hemorrhage':[1], 
                 'Subdural hemorrhage':[2],
                 'Intraventricular hemorrhage':[4], 
                 'Subarachnoid hemorrhage':[3], 
                 'Contusion and laceration':[5], 
                 'Cerebral hernia':[6],
                 'Malacia':[9], 
                 'Subdural effusion':[12],  
                 'Cerebral edema':[10], 
                 ###'Hydrocephalus':[39],
                 'Infarction':[7,8],
                 'Tumor': [199] 
                 }
        '''
        groups = {'脑实质出血':[0],
                  '硬膜外出血':[1], 
                 '硬膜下出血':[2],
                 '脑室内出血':[4], 
                 '蛛网膜下腔出血':[3], 
                 '脑挫裂伤':[5], 
                 '脑疝':[6],
                 '软化灶':[9], 
                 '硬膜下积液':[12], 
                 '脑水肿':[10], 
                 #'Hydrocephalus':[39],
                 '脑梗死':[7,8],
                 '肿瘤': [199] 
                 }
        '''
    group = {}
    for g in groups:
        for d in groups[g]:
            group[d] = g
    scores = {}
    load_scores_single_region('prediction/test_prospective_85/scores_il.txt', scores, il = True)
    print(len(scores))
    abnormal_scores = []
    abnormal_scores0 = []
    abnormal_scores1 = []
    abnormal_scores2 = []
    abnormal_scores3 = []
    with open('Y:/project/EHR/disease.json' if not doctor else 'Y:/project/EHR/disease_doctor2.json', 'r') as fp:
        disease = json.load(fp)
    print(len(disease))
    with open('Y:/project/EHR/report_desc_p2.json', 'r') as fp:
        report_desc = json.load(fp)
    with open('Y:/prospective/cor_reports.json', 'r') as fp:
        cor_report = json.load(fp)
    with open('disease_good_series.json', 'r') as fp:
        good_series = json.load(fp)
    debug = False
    if debug:
        good_series = random.sample(list(good_series.keys()), 100)
    
    l = {x:[] for x in range(200)} #score list
    gl = {x:[] for x in groups} #group score list
    for x in disease:
        post_surgery = '术后' in report_desc[cor_report[x]][1]
        for s in report_desc[cor_report[x]][0]:
            if '术后' in s:
                post_surgery = True
        tumor = False
        s = report_desc[cor_report[x]][1]
        if ('瘤' in s or '转移' in s) and '骨瘤' not in s and '脂肪瘤' not in s\
        and not '未见明确肿瘤' in s and not '未见明确转移' in s and not '血管瘤' in s\
        and not '骨髓瘤' in s and not '动脉瘤' in s and not '强化灶' in s \
        and not '左侧顶骨肿瘤性病变' in s:
            tumor = True
        
        if post_surgery and ignore_post_surgery:
            continue
        if x not in good_series:
            continue
        if '-1_0_prospective_'+x not in scores:
            continue
        sick_group = []
        #if 12 in disease[x]:
        #    print(x, report_desc[x.split('_')[0]], scores['-1_2_prospective_'+x][1])
        if tumor:
            disease[x].append(199)
        for d in disease[x]:
            if d in group and group[d] not in sick_group:
                sick_group.append(group[d])
            #l[d].append((1,scores['-1_0_prospective_'+x][1]))
            #l[d].append((1,max([scores['-1_%d_prospective_'%i+x][1] for i in range(4)])))
            l[d].append((1,max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,4)])))
        #if 'Cerebral edema' in sick_group:
        if tumor:
            print(x, report_desc[cor_report[x]], scores['-1_2_prospective_'+x][1])
        for g in groups:
            if g in sick_group:
                gl[g].append((1,max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,4)])))
        if len(disease[x])==0:
            for d in range(200):
                #l[d].append((0,scores['-1_0_prospective_'+x][1]))
                #l[d].append((0,max([scores['-1_%d_prospective_'%i+x][1] for i in range(4)])))
                l[d].append((0,max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,4)])))
            for g in groups:
                gl[g].append((0,max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,4)])))
        abnormal_scores.append((max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,4)]), sick_group))
        abnormal_scores0.append((max([scores['-1_%d_prospective_'%i+x][1] for i in range(0,1)]), sick_group))
        abnormal_scores1.append((max([scores['-1_%d_prospective_'%i+x][1] for i in range(1,2)]), sick_group))
        abnormal_scores2.append((max([scores['-1_%d_prospective_'%i+x][1] for i in range(2,3)]), sick_group))
        abnormal_scores3.append((max([scores['-1_%d_prospective_'%i+x][1] for i in range(3,4)]), sick_group))
    return
    print(len(abnormal_scores))
    colors = [x for x in mcolors.TABLEAU_COLORS]
    colors += ['blue', 'gold', 'rosybrown']
    union = 'Any'
    temp = list(groups.keys())+[union]
    colors = {temp[i]:colors[i] for i in range(len(temp))}
    ls = ['-','--']*10
    ls = {temp[i]:ls[i] for i in range(len(temp))}
    
    #fig, ax = plt.subplots(1, 1, figsize = (6,6))
    #draw_screen_curve(abnormal_scores, [x for x in groups], ax)
    #fig, ax = plt.subplots(1, 1, figsize = (6,6))
    #draw_screen_curve_growth(abnormal_scores, [x for x in groups], ax)
    
    draw = True
    save = True
    result = [None for i in range(5)]
    fig, axs = plt.subplots(4, 1, figsize = (12,24))
    
    result[1] = draw_screen_bar(abnormal_scores0, [x for x in groups], colors, axs[0], draw = draw, sort_index=sort_index, draw_recall=draw_recall)
    result[2] = draw_screen_bar(abnormal_scores1, [x for x in groups], colors, axs[1], draw = draw, sort_index=sort_index, draw_recall=draw_recall)
    result[3] = draw_screen_bar(abnormal_scores2, [x for x in groups], colors, axs[2], draw = draw, sort_index=sort_index, draw_recall=draw_recall)
    result[4] = draw_screen_bar(abnormal_scores3, [x for x in groups], colors, axs[3], draw = draw, sort_index=sort_index, draw_recall=draw_recall)
    if draw and not debug and save:
        plt.savefig('screen bar (separate)%s%s%d.svg'%('no surgery' if ignore_post_surgery else '', 
                    ' ' if not doctor else '_doctor',
                    sort_index), bbox_inches='tight')
    
    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(2,1,1) #要share坐标轴的时候，又能在两个子图上都显示坐标轴，只能一个一个add。
    result[0] = draw_screen_bar(abnormal_scores, [x for x in groups], colors, ax, draw = draw, sort_index=sort_index, draw_recall=draw_recall)
    
    recall75 = {d:[] for d in list(groups.keys())+[union]}
    
    recall50 = {d:[] for d in list(groups.keys())+[union]}
    
    best_type = {d:(0,(1,1,1,1)) for d in list(groups.keys())+[union]} #best_type[disease] = (type, ranking, other ranking)
    #rank = {i:{} for i in range(5)} #输出给R语言做wilcox检验的。 rank[i]是用i异常作为分数的效果，
    
    for i in range(5):
        for j in range(len(result[i])): #对每一种疾病
            temp = result[i][j] #temp:[disease, rank list, y, mean, I] 其中y和mean是各种指标(四分位点、平均数)的值和bootstrap的mean
            d = temp[0]
            r = np.sum(temp[1][:int(0.75*len(temp[1]))])/np.sum(temp[1])
            mean, I = bootstrap_ranking(temp[1], e = np.array(r))
            recall75[d].append((r, mean, I))
            
            r = np.sum(temp[1][:int(0.5*len(temp[1]))])/np.sum(temp[1])
            recall50[d].append(r)
            if i>0 and temp[2][sort_index]<best_type[d][1][sort_index]: #50%recall所需要查看的百分比
                best_type[d] = (i, temp[2], temp[3], temp[4])
    
    
    for j in range(len(result[0])):
        print(result[0][j][0]) #disease
        
    #for i in range(len(result[0][0][1])): 
    #    writer.writerow([result[0][j][1][i] for j in range(len(result[0]))])
    '''
    table = pt.PrettyTable(border = False)
    for d in recall75:
        table.add_row([d]+recall75[d])
    print(table)
    table = pt.PrettyTable(border = False)
    for d in recall50:
        table.add_row([d]+recall50[d])
    print(table)
    '''
    abnormality_types = ['Intra-hyper', 'Extra-hyper', 'Intra-hypo', 'Extra-hypo']
    for d in groups:
        t = best_type[d][0]-1
        print(d+' & '+abnormality_types[t], end='&')
        #print('%s'%(ps(best_type[d][1][1])), end='&')
        #print('%s'%(ps(best_type[d][1][2])), end='&')
        
        #print('%s (%s-%s)'%(ps(best_type[d][1][1]), ps(best_type[d][2][1]-best_type[d][3][1]), ps(best_type[d][2][1]+best_type[d][3][1])), end='&')
        print('%s (%s-%s)'%(ps(best_type[d][1][2]), ps(best_type[d][2][2]-best_type[d][3][2]), ps(best_type[d][2][2]+best_type[d][3][2])), end='&')
        #print(ps(recall50[d][t+1]), end='&')
        print('%s (%s-%s)'%(ps(recall75[d][t+1][0]), ps(recall75[d][t+1][1]-recall75[d][t+1][2]), ps(recall75[d][t+1][1]+recall75[d][t+1][2])), '\\\\')
        
        #print(d, best_type[d], recall75[d][best_type[d][0]], recall50[d][best_type[d][0]])
    
    ax = fig.add_subplot(2,1,2, sharex=ax)
    result[0].reverse()
    for d in [x[0] for x in result[0]]: #为了和使用所有异常筛查时，疾病的有效率顺序保持一致。不用groups。
        y = []
        for i in range(5):
            result_d = list(filter(lambda x:x[0]==d, result[i]))[0] #因为result[i]中不同疾病排序过了，现在找出对应的疾病
            y.append(result_d[2][sort_index])
        markers = ['<', 'o', '>', 's']
        ax.plot(y, 4-np.arange(5), '-%s'%markers[sort_index], label = d, color = colors[d], linestyle = ls[d])
    ax.set_xlim(0,1)
    ax.set_yticks([4,3,2,1,0])
    
    ax.set_yticklabels(['All abnormality',
                        'Intraparenchymal hyperdensity', 
                        'Extraparenchymal hyperdensity', 
                        'Intraparenchymal hypodensity', 
                        'Extraparenchymal hypodensity'])
    '''
    ax.set_yticklabels(['所有异常',
                        '脑实质内高密度', 
                        '脑实质外高密度', 
                        '脑实质内低密度', 
                        '脑实质外低密度'])
    '''
    ax.set_xticks(np.arange(0,1.01,0.1))
    ax.set_xticklabels(np.arange(0,101,10, dtype=np.uint8))
    if sort_index==1:
        #ax.set_xlabel('Medium ranking (%)')
        ax.set_xlabel('异常排名中位数 (%)')
    elif sort_index==2:
        ax.set_xlabel('Upper quartile ranking (%)')
    
    #ax.set_xlabel('Ranking at recall %.2f (%%)'%(0.25*(sort_index+1))) #sort_index=3，即用平均值排序时不对
    legend = ax.legend(loc = 'upper right', fontsize='medium')
    #legend.legendHandles[0]._legmarker.set_alpha(0.5) #https://www.tutorialspoint.com/how-to-set-legend-marker-size-and-alpha-in-matplotlib
    if draw and not debug and save:
        plt.savefig('screen%s%s%d.svg'%(' no surgery' if ignore_post_surgery else '', 
                                        ' ' if not doctor else '_doctor',
                                        sort_index), bbox_inches='tight')
    
    '''
    #以下是各种疾病vs正常CT二分类的结果，不用
    with open('Y:/prospective/sick_name.json', 'r', encoding='utf-8') as fp:
        sick_name = json.load(fp)
    
    sick_descrip_list = []
    for x in sick_descrip_dict:
        sick_descrip_list.append(x)
    
    print(len(sick_descrip_dict))
    print(len(sick_name))
    
    #for i in range(len(sick_name)):
    #    if str(i) in sick_name:
    #        print(i, len(l[i])-len(l[199]), sick_name[str(i)], draw_roc(l[i], draw = False))
    aucs = []
    for g in groups:
        #print(g, draw_roc(gl[g], draw = False))
        auc = draw_roc(gl[g], draw = False)
        mean,I = bootstrap(gl[g], 5, auc)
        p = sum([x[0] for x in gl[g]])
        print('%s & %.2f & %.2f & (%.2f,%.2f) \\\\'%(g, p/len(gl[g])*100, auc*100, (mean-I)*100, (mean+I)*100))
        aucs.append(auc)
    print(np.mean(np.asarray(aucs)))
    '''
def output_latex_table(prefix = ''):
    def load_ratio(prefix):
        ratio_label = np.loadtxt('paper/%spositive_ratio.txt'%prefix)
        ratio_type = np.loadtxt('paper/%spositive_ratio_type.txt'%prefix)
        ratio_organ = np.loadtxt('paper/%spositive_ratio_organ.txt'%prefix)
        ratio_whole = np.loadtxt('paper/%spositive_ratio_whole.txt'%prefix)
        ratio_label[15,1] = 0
        ratio_label[16,1] = 0
        ratio_label[15,3] = 0
        ratio_label[16,3] = 0
        ratio_label[13,0] = 0
        ratio_label[14,0] = 0
        ratio_label[0,1] = 0
        ratio_label[1,1] = 0
        return ratio_label, ratio_type, ratio_organ, ratio_whole
    
    if type(prefix)==list:
        ratio_label, ratio_type, ratio_organ, ratio_whole = load_ratio(prefix[0])
        ratio_label2, ratio_type2, ratio_organ2, ratio_whole2 = load_ratio(prefix[1])
    else:
        ratio_label, ratio_type, ratio_organ, ratio_whole = load_ratio(prefix)
    '''
    organs = ['left centrum semiovale', 'right centrum semiovale', 'left temporal lobe', 'right temporal lobe',
              'left cerebellum', 'right cerebellum', '', '', 'left frontal lobe', 'right frontal lobe',
              'left parietal lobe', 'right parietal lobe', '', 
              'left lateral ventricle', 'right lateral ventricle', 'left basal ganglia area', 'right basal ganglia area',
              'brain stem', 'left occipital lobe', 'right occipital lobe']
    '''
    print(len(organs))
    for i in organs:
        print('%s '%organ_names[i], end = '')
        for j in range(4):
            print('& %.1f '%(ratio_label[i,j]*100), end='')
        print('& %.1f '%(ratio_organ[i]*100), end='')
        if type(prefix)==list: #两个数据集的
            for j in range(4):
                print('& %.1f '%(ratio_label2[i,j]*100), end='')
            print('& %.1f '%(ratio_organ2[i]*100), end='')
        print('\\\\')
    print('Any ', end='')
    for i in range(4):
        print('& %.1f '%(ratio_type[i]*100), end='')
    print('& %.1f '%(ratio_whole*100), end='')
    if type(prefix)==list:
        for i in range(4):
            print('& %.1f '%(ratio_type2[i]*100), end='')
        print('& %.1f '%(ratio_whole2*100), end='')

def temp():
    size = [1000,2000,7224]
    auc = np.array([[0.997,0.9918,0.9933,0.9947],
                    [0.9991,0.996,0.9894,0.9975],
                    [0.9988,0.9948,0.9930,0.9954]])
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(4):
        ax.plot(size, auc[:,i], '-o', label=labels[i])
    ax.grid(axis='y', zorder=0)
    ax.set_xticks(size)
    ax.legend(loc = 'lower right', fontsize='large')
    ax.set_xlabel('NLP Training size', fontsize='large')
    ax.set_ylabel('NLP micro AUC', fontsize='large')
    
def CT_auc_one_score(model, organs, data_file = None, index_file = None, prospective=False):
    if prospective:
        l = load_scores('prediction/test_prospective_%s'%model, organs, data_file=data_file, index_file=index_file)
    else:
        l = load_scores('prediction/test_multitask%s'%model, organs, data_file=data_file, index_file=index_file)
    print(len(l[(organs[0],0)]))
    
    temp = []
    for i in range(4):
        for j in organs:
            temp += l[(j,i)]
    auc = draw_roc(temp, draw = False)
    mean, I = bootstrap2(temp, 1000, auc)
    print('mean', auc, '(%.3f,%.3f)'%(mean-I, mean+I))

#organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
organs = [8,9,2,3,10,11,18,19,0,1,15,16,13,14,4,5,17,]

organ_order = []
organ_group = [[8,9], [2,3], [10,11], [18,19], [0,1], [15,16], [13,14], [4,5], [17]]
organ_group_name = ['Frontal lobe     ', 
                    'Temporal lobe    ', 
                    'Parietal lobe    ', 
                    'Occipital lobe   ',
                    'Centrum ovale    ', 
                    'Basal ganglia    ',
                    'Lateral ventricle', 
                    'Cerebellum       ',
                    'Brain stem       ', 
                    ]
organ_group_name = ['额叶      ', 
                    '颞叶      ', 
                    '顶叶      ', 
                    '枕叶      ',
                    '半卵圆中心', 
                    '基底节区  ',
                    '侧脑室    ', 
                    '小脑      ',
                    '脑干      ', 
                    ]
side_group = [[0,2,4,8,10,15,18],[1,3,5,9,11,16,19]]

organ_names = ['L centrum ovale', 'R centrum ovale', 
               'L temporal lobe    ', 'R temporal lobe    ',
               'L cerebellum       ', 'R cerebellum       ', '', '', 
               'L frontal lobe     ', 'R frontal lobe     ',
               'L parietal lobe    ', 'R parietal lobe    ', '', 
               'L lateral ventricle', 'R lateral ventricle', 
               'L basal ganglia', 'R basal ganglia',
               'Brain stem            ', 
               'L occipital lobe   ', 'R occipital lobe   ']
'''
organ_names = ['左半卵圆中心', '右半卵圆中心', 
               '左颞叶  ', '右颞叶  ',
               '左小脑  ', '右小脑  ', '', '', 
               '左额叶  ', '右额叶  ',
               '左顶叶  ', '右顶叶  ', '', 
               '左侧脑室  ', '右侧脑室  ', 
               '左基底节区', '右基底节区',
               '脑干    ', 
               '左枕叶  ', '右枕叶  ']
'''

labels = ['Intra-hyper', 'Extra-hyper', 'Intra-hypo ', 'Extra-hypo ']
#labels = ['脑内高密度', '脑外高密度', '脑内低密度', '脑外低密度']
#plt.rcParams['font.sans-serif'] = ['SimHei']


#CT_auc_one_score('85', organs)
#CT_auc_one_score('85', organs, data_file = '//166.111.81.48/liuaohan2/co-training/prospective_160_80.h5', 
#            index_file='//166.111.81.48/liuaohan2/co-training/prospective_good6_1500.pkl', prospective=True)

#draw_CT_auc('85', organs, n_bootstrap=20)
#CT_AUC_latex('85', organs, n_bootstrap=1000)
#CT_AUC_latex('85.2', organs, n_bootstrap=1000)
#CT_AUC_latex('85.3', organs, n_bootstrap=1000)
#CT_AUC_latex('85.4', organs, n_bootstrap=1000)

#draw_NLP_auc(label_file = 'NLP_baseline_soft') #retrospective
#draw_NLP_auc(label_file = 'label_soft_1500') #prospective

#draw_cq500_auc()
#draw_auc_train_size(n_bootstrap = 1000)
#draw_single_task(1000)

draw_CT_compare(n_bootstrap = 1000) #region-level和image-level MIL对比

#draw_CT_auc('85.100', organs)
#draw_CT_auc('85.101', organs)
#draw_auc_train_size(NLP = True, n_bootstrap=1000)

#temp()

#detect_disease()
#detect_disease(ignore_post_surgery = True)
#detect_disease(ignore_post_surgery = True, sort_index=2)
#detect_disease(ignore_post_surgery = True, sort_index=3)

#detect_disease(ignore_post_surgery = True, sort_index=3, doctor = True)
#detect_disease(ignore_post_surgery = True, sort_index=1, doctor = True, draw_recall=False) #revise版本是这个
#detect_disease(ignore_post_surgery = True, sort_index=2, doctor = True) #this one

'''
#试一下scipy的bootstrap，失败了
score = [(0,0.1),(1,0.7),(0,0.8)]
print(draw_roc(score, draw=False))
print(scipy.stats.bootstrap((np.array(score),), lambda x,axis:draw_roc(x,draw=False), method='percentile'))

data = (np.array([x[0] for x in score]), np.array([x[1] for x in score]))
print(AUC(*data))
print(data)
print(scipy.stats.bootstrap(data, AUC, n_resamples='100', method='percentile'))
'''

'''
draw_CT_auc('85', organs, data_file = '//166.111.81.48/liuaohan2/co-training/prospective_160_80.h5', 
            index_file='//166.111.81.48/liuaohan2/co-training/prospective_good6_1500.pkl', prospective=True,
            n_bootstrap = 1000)
'''
#CT_AUC_latex('85', organs, data_file = '//166.111.81.48/liuaohan2/co-training/prospective_160_80.h5', 
#            index_file='//166.111.81.48/liuaohan2/co-training/prospective_good6_1500.pkl',prospective = True)

#output_latex_table('NLP_') #NLP预测，回顾测试集上
#output_latex_table('retrospective_')
#output_latex_table('prospective_')
#output_latex_table(['retrospective_', 'prospective_'])

'''
with open('/home/heyuwei/CT/all_sick_classification_pro_20211025/valid.json', 'r') as fp:
    file = json.load(fp)
for x in file:
    if 107 in x['label']:
        print(x['report'])
'''

#draw_CT_auc('90', organs, n_bootstrap=1000) #分类模型
#draw_CT_auc('85.1', organs, n_bootstrap=1000) #普通MIL
#draw_CT_auc('91.2', organs, n_bootstrap=1000) #spatial attention
#draw_CT_auc('92', organs, n_bootstrap=1000) #静态MIL
#draw_CT_auc('85.110', organs, n_bootstrap=1000) #不使用discretizer生成的label，只使用手标的silver-standard label训练
#draw_CT_auc('85.111', organs, n_bootstrap=1000) #其他脑区分割标注训练的
#draw_CT_auc('85.112', organs, n_bootstrap=1000) #其他脑区分割标注训练的
#draw_CT_auc('85.113', organs, n_bootstrap=1000) #其他脑区分割标注训练的
'''
auc1 = draw_CT_auc('85', organs, n_bootstrap=100, save=False) 
auc2 = draw_CT_auc('85.111', organs, n_bootstrap=100, save=False) 
for i in range(4):
    x = ttest_ind(auc1[i], auc2[i])
    print(x)
'''

#reviewer问假阳率和假阴率
scores = {}
#l = load_scores('prediction/test_multitask85', ['il']) #retrospective
#l = load_scores('prediction/test_prospective_85', ['il'], data_file = '//166.111.81.48/liuaohan2/co-training/prospective_160_80.h5', 
#            index_file='//166.111.81.48/liuaohan2/co-training/prospective_good6_1500.pkl')
l = load_scores('prediction/test_prospective_85', ['il'], data_file = '/data/liuaohan/co-training/prospective_160_80.h5', 
            index_file='/data/liuaohan/co-training/prospective_good6_1500.pkl')

for a in range(4):
    print()
    scores = l[(-1,a)]
    scores.sort(key = lambda x:x[1], reverse=True)
    tp, fp = 0, 0
    t = 0.5
    '''
    for x in scores:
        if x[0]:
            tp += 1
        else:
            fp += 1
        if tp*2<fp:
            t = x[1]
            print(tp, fp)
            break
    '''
    print('t', t)
    tn, tp, fp, fn = 0, 0, 0, 0
    for x in l[(-1,a)]:
        if x[0]==0 and x[1]<t:
            tn += 1
        if x[0]==0 and x[1]>t:
            fp += 1
        if x[0]==1 and x[1]<t:
            fn += 1
        if x[0]==1 and x[1]>t:
            tp += 1
    #print('positive', positive)
    #print('negative', negative)
    print('false positive', fp/(tn+fp))
    print('false negative', fn/(tp+fn))
    print('fp', fp)
    print('fn', fn)
