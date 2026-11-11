"""Reproducible paired benchmarks. Run after experiment.py."""
import json, random, statistics, time, platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import astar,tree_search
ROOT=Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
MODES=[('Dijkstra',False,False),('A* small-g',True,False),('A* large-g',True,True)]
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
fig,axs=plt.subplots(2,2,figsize=(10,6.2),layout='constrained')
colors=['#788596','#287bb5','#e07a35']
for (label,_,_),color in zip(MODES,colors):
    rr=[r for r in summary if r['mode']==label]
    axs[0,0].errorbar([r['n'] for r in rr],[r['expanded'] for r in rr],yerr=[r['sd'] for r in rr],marker='o',capsize=3,label=label,color=color)
    axs[0,1].plot([r['n'] for r in rr],[r['ms'] for r in rr],marker='o',label=label,color=color)
axs[0,0].set(title='(a) Maze search: mean +/- SD (20 maps)',xlabel='Grid side length',ylabel='Expanded nodes'); axs[0,0].legend(fontsize=8)
axs[0,1].set(title='(b) Paired runtime benchmark',xlabel='Grid side length',ylabel='Mean of per-map median (ms)')
for key,label in [('minimax','Minimax'),('ab','Alpha-beta')]:
    axs[1,0].plot(range(2,8),[statistics.mean(r[key] for r in trees if r['depth']==d) for d in range(2,8)],marker='o',label=label)
axs[1,0].set(title='(c) Ternary trees: 20 seeds / depth',xlabel='Depth',ylabel='Evaluated leaves',yscale='log'); axs[1,0].legend()
r=json.loads((ROOT/'results.json').read_text()); x=np.arange(2)
for i,(ab,order,label) in enumerate([(False,'natural','Minimax'),(True,'natural','AB natural'),(True,'gain','AB gain order')]):
    vals=[r['game'][f'H={d},ab={ab},{order}']['nodes'] for d in [2,4]]
    bars=axs[1,1].bar(x+(i-1)*.25,vals,width=.24,label=label,color=colors[i]); axs[1,1].bar_label(bars,fontsize=8)
axs[1,1].set(title='(d) Treasure game (root included)',xticks=x,xticklabels=['H=2','H=4'],ylabel='Visited nodes',ylim=(0,80)); axs[1,1].legend(fontsize=8)
for ax in axs.flat: ax.grid(axis='y',alpha=.18)
fig.savefig(ROOT/'performance.pdf'); fig.savefig(ROOT/'performance.png',dpi=180); plt.close(fig)
grid,traces=representative; fig,axs=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
for ax,(label,trace,path) in zip(axs,traces):
    a=np.zeros((25,25)); a[:]=np.nan
    for k,(r,c) in enumerate(trace): a[r,c]=(k+1)/len(trace)
    ax.imshow([[0 if v=='X' else 1 for v in row] for row in grid],cmap='gray',vmin=0,vmax=1)
    ax.imshow(np.ma.masked_invalid(a),cmap='Blues',vmin=0,vmax=1,alpha=.85)
    ax.plot([p[1] for p in path],[p[0] for p in path],color='#e45a3c',lw=1.5)
    ax.set(title=f'{label}: {len(trace)} nodes',xticks=[],yticks=[])
fig.savefig(ROOT/'search_regions.pdf'); fig.savefig(ROOT/'search_regions.png',dpi=180)
print(json.dumps(summary)); print('Tree depth 7:',statistics.mean(r['ab'] for r in trees if r['depth']==7),'vs',3**7)
