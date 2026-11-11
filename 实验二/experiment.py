import heapq, itertools, json
from pathlib import Path

def astar(grid, start, goal, heuristic=True, terminal_zero=False, large_g=False, trace=None):
    def cost(p):
        v=grid[p[0]][p[1]]
        return 0 if terminal_zero and p==goal else (v if isinstance(v,int) else 1)
    minimum=min([cost((r,c)) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c]!='X' and (r,c)!=goal]+[1])
    def h(p):
        d=abs(p[0]-goal[0])+abs(p[1]-goal[1])
        return minimum*max(0,d-(1 if terminal_zero else 0)) if heuristic else 0
    q=[(h(start),0,0,start)]; dist={start:0}; parent={}; expanded=0
    while q:
        _,_,g,p=heapq.heappop(q)
        if g!=dist[p]: continue
        expanded+=1
        if trace is not None: trace.append(p)
        if p==goal:
            path=[p]
            while p in parent: p=parent[p]; path.append(p)
            return g,path[::-1],expanded
        for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            n=(p[0]+dr,p[1]+dc)
            if 0<=n[0]<len(grid) and 0<=n[1]<len(grid[0]) and grid[n[0]][n[1]]!='X':
                ng=g+cost(n)
                if ng<dist.get(n,float('inf')):
                    dist[n]=ng; parent[n]=p; heapq.heappush(q,(ng+h(n),-ng if large_g else ng,ng,n))
    return -1,[],expanded

def tree_search(tree, pruning, reverse=False):
    counts=[0,0]
    def visit(t,maxi,a,b):
        counts[0]+=1
        if isinstance(t,int): counts[1]+=1; return t
        v=float('-inf') if maxi else float('inf')
        for child in (t[::-1] if reverse else t):
            x=visit(child,not maxi,a,b); v=max(v,x) if maxi else min(v,x)
            if maxi: a=max(a,v)
            else: b=min(b,v)
            if pruning and a>=b: break
        return v
    return visit(tree,True,float('-inf'),float('inf')),counts

def game(D,H,pruning,order='natural',cache=False):
    counts=[0,0,0]; memo={}; values=[9,6,7,7]
    def visit(pa,pb,remaining,turn,depth,a,b):
        counts[0]+=1
        key=(pa,pb,remaining,turn,depth)
        if cache and key in memo: counts[2]+=1; return memo[key],None
        if depth==0 or not remaining: counts[1]+=1; return 0,None
        moves=list(remaining); pos=pa if turn==0 else pb
        if order=='reverse': moves.reverse()
        if order=='gain': moves.sort(key=lambda t:values[t-2]-D[pos][t],reverse=True)
        best=float('-inf') if turn==0 else float('inf'); target=None; cut=False
        for t in moves:
            gain=values[t-2]-D[pos][t]
            child,_=visit(t if turn==0 else pa,t if turn else pb,tuple(x for x in remaining if x!=t),1-turn,depth-1,a-gain if turn==0 else a+gain,b-gain if turn==0 else b+gain)
            v=child+(gain if turn==0 else -gain)
            if (turn==0 and v>best) or (turn==1 and v<best): best,target=v,t
            if turn==0: a=max(a,best)
            else: b=min(b,best)
            if pruning and a>=b: cut=True; break
        if cache and not pruning: memo[key]=best
        return best,target
    v,t=visit(0,1,(2,3,4,5),0,H,float('-inf'),float('inf'))
    return dict(value=v,target='T'+str(t-1),nodes=counts[0],leaves=counts[1],hits=counts[2])

def main():
    maze=[list(s) for s in ['S..X.','.X...','.X.X.','...XE','XX...']]
    weighted=[['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
    grid=[['A',1,1,'X',2,1,'T1'],[1,'X',6,'X',1,'X',1],[1,1,'T2',1,1,1,1],['X',1,'X',8,'X',1,'X'],[1,1,1,1,'T3',1,1],[1,'X',9,'X',1,'X',1],['T4',1,1,'X',1,1,'B']]
    points=[(0,0),(6,6),(0,6),(2,2),(4,4),(6,0)]
    D=[[astar(grid,a,b)[0] for b in points] for a in points]
    results={'maze':{str(h):astar(maze,(0,0),(3,4),h) for h in [False,True]},'weighted':{str(h):astar(weighted,(0,0),(4,4),h,True) for h in [False,True]},'matrix':D,'tree':{},'game':{},'extra':{}}
    tree=[[[3,5],[6,9]],[[6,9],[1,2]]]
    for rev in [False,True]:
        for ab in [False,True]: results['tree'][f'reverse={rev},ab={ab}']=tree_search(tree,ab,rev)
    for H in [2,4]:
        for ab in [False,True]:
            for order in ['natural','reverse','gain']: results['game'][f'H={H},ab={ab},{order}']=game(D,H,ab,order)
    for n in [10,30]:
        g=[[1]*n for _ in range(n)]
        results['extra'][str(n)]={str(h):astar(g,(0,0),(n-1,n-1),h)[::2] for h in [False,True]}
        results['extra'][str(n)]['large_g']=astar(g,(0,0),(n-1,n-1),True,large_g=True)[::2]
    results['cache']=game(D,4,False,cache=True)
    assert results['game']['H=2,ab=True,natural']['value']==-1
    assert results['game']['H=4,ab=True,natural']['value']==3
    assert astar([['S','X','E']],(0,0),(0,2))[0]==-1
    assert astar([[1]],(0,0),(0,0))[0]==0
    Path(__file__).with_name('results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__': main()
