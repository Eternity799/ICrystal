"""B3: fixed startup, free endpoint, full PBM, unchanged MPC scoring."""
from pathlib import Path
import sys
import copy
import numpy as np

from .controller import TrialProblem


class NodeProblem(TrialProblem):
    def __init__(self,predict,config,node_minutes,start_temperature,quality_min=None,quality_max=None):
        self.nodes=np.asarray(node_minutes,dtype=float)
        if (self.nodes.ndim!=1 or len(self.nodes)<2 or self.nodes[0]!=10 or self.nodes[-1]!=180
            or not np.isfinite(self.nodes).all() or np.any(np.diff(self.nodes)<=0)
            or np.any(self.nodes!=np.round(self.nodes))):
            raise ValueError('Nodes must be increasing integer minutes from 10 to 180')
        self.start=float(start_temperature)
        self.startup_rate=(config['constraints']['min_temperature_c']-self.start)/180
        super().__init__(predict,self.start+10*self.startup_rate,self.startup_rate,170,
                         config,quality_min,quality_max)
        self.matrix=np.eye(170)

    def curve(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(len(self.nodes)-1,) or not np.isfinite(x).all():
            raise ValueError('One finite temperature per free node required')
        return np.r_[self.start+self.startup_rate*np.arange(10),
                     np.interp(np.arange(10,181),self.nodes,np.r_[self.temperature,x])]

    def evaluate(self,x):
        return super().evaluate(np.diff(self.curve(x))[10:])


class PBMPredictor:
    def __init__(self,cfg,startup):
        from physics.simulation import BatchPlant
        self.plant=BatchPlant.initialize(cfg)
        for t in startup[1:]: self.plant.advance(float(t))
        self.last=None

    def __call__(self,temperatures):
        from physics.metrics import volume_percentiles
        plant=copy.deepcopy(self.plant)
        for temperature in temperatures: plant.advance(float(temperature))
        r=plant.result()
        quality=np.column_stack([r.concentration_kg_kg_water,r.yield_fraction,
            np.asarray([volume_percentiles(n,r.grid) for n in r.counts_per_kg_water])])
        valid=(np.isfinite(quality).all(axis=1)&np.isfinite(r.counts_per_kg_water).all(axis=1)
               &(r.counts_per_kg_water>=0).all(axis=1))
        self.last=dict(time_min=r.time_s/60,temperature_c=r.temperature_c,quality=quality,
            counts_per_kg_water=r.counts_per_kg_water,supersaturation=r.supersaturation_ratio,
            relative_mass_error=r.relative_mass_error,pivots_um=r.grid.pivots_m*1e6)
        return dict(quality=quality[11:],diagnostics=dict(valid=valid[11:]))


def command_margins(curve,config):
    c=config['constraints'];rates=np.diff(curve)
    return np.r_[curve-c['min_temperature_c'],c['max_temperature_c']-curve,
                 rates-c['min_rate_c_min'],c['max_rate_c_min']-rates]


def seed_curves(config,start):
    minutes=np.arange(181)
    linear=np.linspace(start,config['constraints']['min_temperature_c'],181)
    nodes=np.asarray(config['empirical_nodes'],dtype=float)
    if nodes.ndim!=2 or nodes.shape[1]!=2 or nodes[0,0]!=0 or nodes[-1,0]!=180 or np.any(np.diff(nodes[:,0])<=0):
        raise ValueError('Invalid empirical time nodes')
    empirical=np.interp(minutes,nodes[:,0],nodes[:,1])
    if not np.allclose(empirical[:11],linear[:11],atol=1e-9,rtol=0):
        raise ValueError('Empirical startup differs from common MPC startup')
    fast=np.interp(minutes,[0,10,90,180],[start,linear[10],30.,20.])
    return dict(linear=linear,empirical=empirical,fast_slow=fast)
