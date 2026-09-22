"""Regression tests for evaluation isolation, gradients, Gaussian flow and resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import workflow
from diagnostics import GradientMoments, draw_features, gradient_diagnostics, interval_diagnostics, loss_gradients
from gaussian_validation import ExactGaussian, gaussian_mean_velocity, gaussian_metrics
from models.meanflow import MeanFlow
from models.weak_loss import weak_terms
from test_workflow import TINY, StubInception, tiny_pixels

torch.set_num_threads(2)

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.3))
        self.interval=torch.nn.Parameter(torch.tensor(.2))
    def forward(self,x,times,aug_cond=None):
        t,h=times
        return self.weight*x+self.interval*h[:,None,None,None]

def model(config=TINY):
    return MeanFlow(TinyNet,workflow.model_args(workflow.validate_config(config)),{})

class Checks(unittest.TestCase):
    def test_fm_condition_is_zero_but_integration_step_is_nonzero(self):
        m=model(); noise=torch.ones(2,3,8,8)
        class Field(torch.nn.Module):
            def forward(self,x,times,aug_cond=None):
                t,h=times
                self.last=h
                return x+100*h[:,None,None,None]
        f=Field()
        out=m.sample(noise.shape,net=f,device='cpu',num_steps=4,initial_noise=noise,sampler='fm_euler')
        torch.testing.assert_close(out,noise*.75**4)
        self.assertEqual(float(f.last.abs().max()),0.)
        self.assertFalse(torch.allclose(out,m.sample(noise.shape,net=f,device='cpu',num_steps=4,initial_noise=noise)))

    def test_multiple_emas_and_initial_mass_removal(self):
        m=model(dict(TINY,ema_decay=.9,ema_decays=[.8]))
        origin=copy.deepcopy(m.state_dict())
        with torch.no_grad(): m.net.weight.fill_(2.)
        for _ in range(32): m.update_ema()
        torch.testing.assert_close(m.net_ema.weight,.9**32*origin['net.weight']+(1-.9**32)*2)
        self.assertIs(workflow.select_weights(m,'ema1'),m.net_ema1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'initial.pt'
            torch.save(dict(format=workflow.FORMAT,step=0,identity='same',model=origin),path)
            net=workflow.select_weights(m,'ema_noinit',dict(step=32,identity='same'),path)
            torch.testing.assert_close(net.weight,torch.tensor(2.))
        with self.assertRaises(ValueError): workflow.select_weights(m,'ema2')
        with self.assertRaises(ValueError): workflow.select_weights(m,'ema_noinit')

    def test_gradient_moments_bias_correction(self):
        samples=torch.tensor([[1.,2.],[3.,4.],[-2.,5.]])
        stats=GradientMoments()
        for x in samples: stats.update(x)
        result=stats.summary(); variance=float(samples.double().var(0).sum())
        self.assertAlmostEqual(result['noise_rms']**2,variance)
        self.assertAlmostEqual(result['signal_squared_unbiased'],float(samples.double().mean(0).square().sum())-variance/3)

    def test_separate_gradients_sum_to_total_and_pair_precision(self):
        cfg=workflow.validate_config(TINY); m=model(cfg); args=workflow.model_args(cfg)
        x=torch.randn(4,3,8,8); features=draw_features(192,8,cfg,20,'cpu')
        _,w32,gd,gw,_=loss_gradients(m.net,x,args,features,9)
        torch.manual_seed(9)
        d,w,_=weak_terms(m.net,x,args,feature_parameters=features)
        direct=torch.autograd.grad(d+w,tuple(m.net.parameters()))
        torch.testing.assert_close(gd+gw,torch.cat([g.flatten() for g in direct]))
        args.weak_fp64=True
        _,w64,_,g64,_=loss_gradients(m.net,x,args,features,9)
        self.assertAlmostEqual(w32,w64,places=6)
        torch.testing.assert_close(gw,g64,atol=1e-6,rtol=1e-5)

    def test_diagnostics_do_not_change_parameters_or_buffers(self):
        cfg=workflow.validate_config(TINY); m=model(cfg); before=copy.deepcopy(m.state_dict())
        payload=dict(config=cfg,step=0)
        with tempfile.TemporaryDirectory() as tmp:
            summary=gradient_diagnostics(m,payload,tiny_pixels(8),tmp,'raw',[4],[8],repeats=3,
                                         resampling=['both','data','features'],precision_repeats=1,device='cpu')
            self.assertEqual(len(summary),3); json.dumps(summary,allow_nan=False)
            result=interval_diagnostics(m,payload,tiny_pixels(8),tmp,batch=4,device='cpu')
            self.assertEqual(len(result['intervals']),40)
            self.assertEqual(len(result['previews']),8)
            self.assertTrue((Path(tmp)/'precision_comparison.csv').exists())
        for k,v in before.items(): torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0)

    def test_gaussian_flow_identity_and_exact_one_step(self):
        z=torch.tensor([.7,-.3],dtype=torch.float64).view(2,1,1,1)
        t=torch.tensor([.6,.9],dtype=torch.float64).view(2,1,1,1)
        r=torch.tensor([.1,.4],dtype=torch.float64).view(2,1,1,1)
        v=gaussian_mean_velocity(z,t,t,.5)
        def F(zz,tt): return (tt-r)*gaussian_mean_velocity(zz,r,tt,.5)
        _,df=torch.func.jvp(F,(z,t),(v,torch.ones_like(t)))
        torch.testing.assert_close(df,v,rtol=1e-10,atol=1e-10)
        metrics=gaussian_metrics(ExactGaussian(.5),.5,1,1,'cpu',count=128)
        self.assertLess(metrics['one_step_map_mse'],1e-12)
        self.assertEqual(metrics['mean_velocity_mse'],0.)

    def test_zero_weight_is_exactly_diagonal_gradient(self):
        cfg=dict(TINY,weak_weight=0.); m=model(cfg); args=workflow.model_args(workflow.validate_config(cfg))
        d,w,_=weak_terms(m.net,torch.randn(4,3,8,8),args)
        gd=torch.autograd.grad(d,tuple(m.net.parameters()),retain_graph=True)
        gt=torch.autograd.grad(d+w,tuple(m.net.parameters()))
        for a,b in zip(gd,gt): torch.testing.assert_close(a,b,rtol=0,atol=0)

class Pipelines(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup); self.pixels=tiny_pixels(12)
        for name,value in [('instantiate_model',lambda args:MeanFlow(TinyNet,args,{})),
                           ('_inception',lambda device:StubInception())]:
            p=patch.object(workflow,name,value);p.start();self.addCleanup(p.stop)
    def train(self,folder,target):
        cfg=dict(TINY,ema_decay=.99,ema_decays=[.9],step_rng=True,snapshot_steps=[2,4])
        return workflow.train(cfg,self.root/folder,'',target,device='cpu',pixels=self.pixels,save_every=2,log_every=2)
    def test_exact_resume_and_saved_snapshots(self):
        self.train('full',4);self.train('split',2);self.train('split',4)
        a=workflow.load_checkpoint(self.root/'full/checkpoint-last.pt')
        b=workflow.load_checkpoint(self.root/'split/checkpoint-last.pt')
        for k in a['model']: torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
        for step in [0,2,4]: self.assertTrue((self.root/'split/checkpoints'/f'step-{step:08d}.pt').exists())
        self.assertEqual(b['config']['ema_decays'],[.9])
    def test_independent_reference_count_partial_batch_and_cache_guards(self):
        self.train('run',2)
        args=dict(checkpoint=self.root/'run/checkpoint-last.pt',output_dir=self.root/'eval',data_root='',
                  nfe_values=[1],num_samples=7,batch_size=3,seed=5,pixels=self.pixels,
                  device='cpu',weights='raw',num_real_samples=12)
        result=workflow.evaluate_fid(**args)
        self.assertEqual(result['results']['1']['n_fake'],7)
        self.assertEqual(result['results']['1']['n_real'],12)
        for changes in [dict(weights='ema'),dict(sampler='fm_euler'),dict(num_real_samples=6)]:
            with self.assertRaises(ValueError): workflow.evaluate_fid(**dict(args,**changes))
        self.assertEqual(workflow.evaluate_fid(**args)['results'],result['results'])

    def test_old_checkpoint_without_new_config_keys_loads(self):
        old=model(TINY)
        config=workflow.validate_config(TINY)
        for key in ('ema_decays','snapshot_steps','step_rng'): config.pop(key)
        path=self.root/'old.pt'
        torch.save(dict(format=workflow.FORMAT,config=config,model=old.state_dict(),step=0),path)
        loaded,_=workflow.load_for_eval(path,'cpu')
        for k,v in old.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[k],v)
        self.assertEqual(workflow.sampling_cost([1,128],5000,50000)['inception_images'],60000)

if __name__=='__main__': unittest.main(verbosity=2)
