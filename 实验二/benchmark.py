# 在同一批地图和博弈树上比较搜索性能。请先运行 experiment.py，再运行本脚本。
import json, random, statistics, time, platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import astar,tree_search
ROOT=Path(__file__).parent
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
MODES=[('Dijkstra',False,False),('A* 小 g 优先',True,False),('A* 大 g 优先',True,True)]
rows=[]; representative=None
for n in [15,25,35,45]:
    accepted=0; seed=0
    while accepted<20:
        rng=random.Random(10000*n+seed); grid=[['X' if rng.random()<.25 else 1 for _ in range(n)] for _ in range(n)]; grid[0][0]=grid[-1][-1]=1
        base=astar(grid,(0,0),(n-1,n-1),False)
        current_seed=seed; seed+=1
        if base[0]<0: continue
        accepted+=1
        for label,h,lg in MODES:
            trace=[]; out=astar(grid,(0,0),(n-1,n-1),h,large_g=lg,trace=trace)
            assert out[0]==base[0]
            samples=[]
            for repeat in range(7):
                t=time.perf_counter_ns(); astar(grid,(0,0),(n-1,n-1),h,large_g=lg); samples.append((time.perf_counter_ns()-t)/1e6)
            rows.append(dict(n=n,seed=10000*n+current_seed,mode=label,cost=out[0],expanded=out[2],ms=statistics.median(samples)))
            if n==25 and accepted==1:
                if representative is None: representative=(grid,[])
                representative[1].append((label,trace,out[1]))
summary=[]
for n in [15,25,35,45]:
    for label,_,_ in MODES:
        rr=[r for r in rows if r['n']==n and r['mode']==label]
        summary.append(dict(n=n,mode=label,expanded=statistics.mean(r['expanded'] for r in rr),sd=statistics.stdev(r['expanded'] for r in rr),ms=statistics.mean(r['ms'] for r in rr)))
trees=[]
for depth in range(2,8):
    for seed in range(20):
        rng=random.Random(depth*1000+seed)
        def build(d): return rng.randrange(-100,101) if d==0 else [build(d-1) for _ in range(3)]
        tree=build(depth); v,c=tree_search(tree,False); w,e=tree_search(tree,True); assert v==w
        trees.append(dict(depth=depth,seed=seed,minimax=c[1],ab=e[1]))
data=dict(environment=platform.python_version(),maps=rows,summary=summary,trees=trees)
(ROOT/'benchmark_results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,2,figsize=(10,5),layout='constrained')
colors=['#788596','#287bb5','#e07a35']
for (label,_,_),color in zip(MODES,colors):
    rr=[r for r in summary if r['mode']==label]
    axs[0,0].errorbar([r['n'] for r in rr],[r['expanded'] for r in rr],yerr=[r['sd'] for r in rr],marker='o',capsize=3,label=label,color=color)
    axs[0,1].plot([r['n'] for r in rr],[r['ms'] for r in rr],marker='o',label=label,color=color)
axs[0,0].set(title='(a) 迷宫：均值与标准差（20 张图）',xlabel='网格边长',ylabel='扩展节点数'); axs[0,0].legend(fontsize=8)
axs[0,1].set(title='(b) 同图配对运行时间',xlabel='网格边长',ylabel='各图耗时中位数的均值（毫秒）')
for key,label in [('minimax','Minimax'),('ab','Alpha-beta')]:
    axs[1,0].plot(range(2,8),[statistics.mean(r[key] for r in trees if r['depth']==d) for d in range(2,8)],marker='o',label=label)
axs[1,0].set(title='(c) 三叉树：每深度 20 个种子',xlabel='搜索深度',ylabel='叶子评估数',yscale='log'); axs[1,0].legend()
r=json.loads((ROOT/'results.json').read_text()); x=np.arange(2)
for i,(ab,order,label) in enumerate([(False,'natural','Minimax'),(True,'natural','剪枝：自然顺序'),(True,'gain','剪枝：收益排序')]):
    vals=[r['game'][f'H={d},ab={ab},{order}']['nodes'] for d in [2,4]]
    bars=axs[1,1].bar(x+(i-1)*.25,vals,width=.24,label=label,color=colors[i]); axs[1,1].bar_label(bars,fontsize=8)
axs[1,1].set(title='(d) 宝藏分配（包含根节点）',xticks=x,xticklabels=['H=2','H=4'],ylabel='访问节点数',ylim=(0,80)); axs[1,1].legend(fontsize=8)
for ax in axs.flat: ax.grid(axis='y',alpha=.18)
fig.savefig(ROOT/'performance.pdf'); fig.savefig(ROOT/'performance.png',dpi=180); plt.close(fig)
grid,traces=representative; fig,axs=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
for ax,(label,trace,path) in zip(axs,traces):
    a=np.zeros((25,25)); a[:]=np.nan
    for k,(r,c) in enumerate(trace): a[r,c]=(k+1)/len(trace)
    ax.imshow([[0 if v=='X' else 1 for v in row] for row in grid],cmap='gray',vmin=0,vmax=1)
    ax.imshow(np.ma.masked_invalid(a),cmap='Blues',vmin=0,vmax=1,alpha=.85)
    ax.plot([p[1] for p in path],[p[0] for p in path],color='#e45a3c',lw=1.5)
    ax.set(title=f'{label}：{len(trace)} 个节点',xticks=[],yticks=[])
fig.savefig(ROOT/'search_regions.pdf'); fig.savefig(ROOT/'search_regions.png',dpi=180)
print(json.dumps(summary)); print('Tree depth 7:',statistics.mean(r['ab'] for r in trees if r['depth']==7),'vs',3**7)



