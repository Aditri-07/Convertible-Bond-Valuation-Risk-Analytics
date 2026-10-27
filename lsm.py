import sys, time, numpy as np
sys.path.insert(0,'.')
from analysis import BASE
from cb_pricer import price_tf, coupon_at_step
def lsm_tf(S0, t, n_paths=50_000, seed=0, deg=3):
    rng=np.random.default_rng(seed); N=t.n_steps; dt=t.maturity/N
    Z=rng.standard_normal((n_paths//2,N)); Z=np.vstack([Z,-Z])          # antithetic
    lnS=np.log(S0)+np.cumsum((t.r_riskfree-0.5*t.vol**2)*dt+t.vol*np.sqrt(dt)*Z,axis=1)
    S=np.exp(np.hstack([np.full((n_paths,1),np.log(S0)),lnS]))
    rf, rr = t.r_riskfree, t.r_riskfree+t.credit_spread
    cpn=np.array([coupon_at_step(k,dt,t) for k in range(N+1)])
    # per-path cash flows split TF-style: equity leg (disc rf) and cash leg (disc risky), valued at current step
    conv=t.conversion_ratio*S[:,N]; red=t.face+cpn[N]
    eq=np.where(conv>red,conv,0.0); cash=np.where(conv>red,0.0,red)
    for k in range(N-1,0,-1):
        eq*=np.exp(-rf*dt); cash=cash*np.exp(-rr*dt)+cpn[k]
        hold=eq+cash; x=S[:,k]/S0
        X=np.vander(x,deg+1); beta=np.linalg.lstsq(X,hold,rcond=None)[0]
        cont=X@beta; cv=t.conversion_ratio*S[:,k]
        ex=cv>cont                                    # decision uses regression, payoff uses realized
        eq=np.where(ex,cv,eq); cash=np.where(ex,0.0,cash)
    v=eq*np.exp(-rf*dt)+cash*np.exp(-rr*dt)+cpn[0]
    return max(v.mean(), t.conversion_ratio*S0), v.std()/np.sqrt(n_paths)
print(f"{'S0':>4} {'lattice':>9} {'LSM':>9} {'SE':>6} {'diff%':>7}")
for S0 in [20,40,50,60,75,100,200]:
    a=time.time(); m,se=lsm_tf(S0,BASE); L=price_tf(S0,BASE)[0]
    print(f"{S0:4d} {L:9.3f} {m:9.3f} {se:6.3f} {100*(m-L)/L:7.3f}  ({time.time()-a:.1f}s)")

# diagnostic: never convert early (hold to maturity) — should equal lattice if early conversion is never optimal
def euro(S0,t,n_paths=50_000,seed=0):
    rng=np.random.default_rng(seed); N=t.n_steps; dt=t.maturity/N; T=t.maturity
    Z=rng.standard_normal(n_paths//2); Z=np.concatenate([Z,-Z])
    ST=S0*np.exp((t.r_riskfree-0.5*t.vol**2)*T+t.vol*np.sqrt(T)*Z)
    cpn=np.array([coupon_at_step(k,dt,t) for k in range(N+1)]); rr=t.r_riskfree+t.credit_spread
    pv_cpn=sum(c*np.exp(-rr*k*dt) for k,c in enumerate(cpn[:N]) if c)
    conv=t.conversion_ratio*ST; red=t.face+cpn[N]
    v=np.where(conv>red,conv*np.exp(-t.r_riskfree*T),red*np.exp(-rr*T))+pv_cpn
    return v.mean(), v.std()/np.sqrt(n_paths)
print("\nhold-to-maturity MC vs lattice")
for S0 in [20,50,75,200]:
    m,se=euro(S0,BASE); L=price_tf(S0,BASE)[0]; print(S0, round(L,3), round(m,3), round(se,3), f"{100*(m-L)/L:.3f}%")
