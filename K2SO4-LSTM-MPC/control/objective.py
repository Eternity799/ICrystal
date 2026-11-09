"""One terminal objective shared by online MPC, offline B3 and reporting."""
import numpy as np

def terminal_components(quality,config):
    if config.get('objective')!='d50_span_priority_v1':
        raise ValueError('Expected d50_span_priority_v1 objective')
    q=np.asarray(quality,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all() or q[5]<0:
        raise ValueError('Invalid terminal quality')
    n=config['normalization'];w=config['weights'];t=config['target']
    return dict(
        d50=float(w['d50_terminal']*((q[3]-t['d50_um'])/n['d50_um'])**2),
        yield_term=float(w['yield_terminal']*((q[1]-t['yield_fraction'])/n['yield_fraction'])**2),
        span=float(w['span_terminal']*q[5]/n['span']),
        span_excess=float(w['span_excess']*(max(q[5]-t['span_upper'],0.)/n['span'])**2))
