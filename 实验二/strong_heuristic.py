"""先把障碍格视为代价为 1 的通路，再计算到终点的最小代价，用它引导原地图上的搜索。"""
import heapq,random,json
from pathlib import Path
from experiment import astar

def trial(n=31,seed=20261112):
    rng=random.Random(seed)
    grid=[['X' if rng.random()<.2 else rng.randint(1,9) for _ in range(n)] for _ in range(n)]
    grid[0][0]=grid[-1][-1]=1
    goal=(n-1,n-1)
    # 从终点反向运行 Dijkstra；原来的障碍格按进入代价 1 处理。
    d={goal:0}; q=[(0,goal)]; pre=0
    while q:
        g,p=heapq.heappop(q)
        if g!=d[p]:continue
        pre+=1
        w=grid[p[0]][p[1]]; w=1 if w=='X' else w
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            v=(p[0]+dr,p[1]+dc)
            if 0<=v[0]<n and 0<=v[1]<n and g+w<d.get(v,10**9):
                d[v]=g+w;heapq.heappush(q,(g+w,v))
    def search():
        best={(0,0):0}; q=[(d[(0,0)],0,(0,0))]; count=0
        while q:
            _,g,p=heapq.heappop(q)
            if g!=best[p]:continue
            count+=1
            if p==goal:return g,count
            for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                v=(p[0]+dr,p[1]+dc)
                if 0<=v[0]<n and 0<=v[1]<n and grid[v[0]][v[1]]!='X':
                    ng=g+grid[v[0]][v[1]]
                    if ng<best.get(v,10**9):best[v]=ng;heapq.heappush(q,(ng+d[v],ng,v))
        return -1,count
    base=astar(grid,(0,0),goal,False); man=astar(grid,(0,0),goal); strong=search()
    assert base[0]==man[0]==strong[0] and base[0]>=0
    return dict(seed=seed,n=n,cost=base[0],dijkstra=base[2],manhattan=man[2],relaxed=strong[1],preprocessing=pre)
if __name__=='__main__':
    r=trial(); print(r); Path(__file__).with_name('strong_results.json').write_text(json.dumps(r,indent=2))
