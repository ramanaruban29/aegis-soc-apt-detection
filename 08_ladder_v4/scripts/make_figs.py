#!/usr/bin/env python3
"""Make the three coloured figures of the JNSM paper from the saved result JSON files.
Usage:  python make_figures.py --roots DIR [DIR ...] --out figs
DIR = any folder that contains (at any depth) the result folders, e.g. the unzipped
cicapt_v4x_results, dapt_v4x_results and chrono_results folders.
Needs: numpy, matplotlib.  Writes fig1_protocol, fig2_ladder, fig3_portshuffle as PDF, EPS and PNG (300 dpi)."""
import argparse, glob, json, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ap=argparse.ArgumentParser(); ap.add_argument('--roots',nargs='+',required=True); ap.add_argument('--out',default='figs')
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
NAMES={'C_row':'cicapt_row_block','C_r5rand':'cicapt_random5_random','C_r5blk':'cicapt_random5_block','C_r5blk_x':'cicapt_random5_block_extra',
 'C_r120blk':'cicapt_random120_block','C_chrono_x':'cicapt_chrono_block_extra','D_row_blk':'results_v4_row_gap5_block','D_r5rand':'results_v4_random_gap5',
 'D_r5blk':'results_v4_random_gap5_block','D_r120':'results_v4_random_gap120','D_chrono':'results_v4_chrono_gap5'}
def find(folder):
    for r in a.roots:
        for d,_,_ in os.walk(r):
            if os.path.basename(d)==folder and glob.glob(d+'/*.json'): return d
    sys.exit(f'folder {folder} not found under {a.roots}')
def load(k): return [json.load(open(f)) for f in sorted(glob.glob(find(NAMES[k])+'/*.json'))]
def merged(k,kx=None):
    R=load(k)
    if kx:
        for r,x in zip(R,load(kx)):
            for m,v in x['res'].items(): r['res'].setdefault(m,v)
    return R
def vals(R,m,rule): return np.array([r['res'][m][rule]['mcc'] if m in r['res'] and rule in r['res'][m] else np.nan for r in R],float)
# colour-blind-safe palette (Okabe-Ito) + different markers, so the figures also work in grey
BLUE,ORANGE,GREEN,VERM,PURPLE,GREY='#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#555555'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.6,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    for e in ('pdf','eps','png'): fig.savefig(f'{a.out}/{name}.{e}',dpi=300,bbox_inches='tight')
    plt.close(fig)
# ---- Fig 1
fig,ax=plt.subplots(figsize=(6.9,3.0)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,44)
def box(x,y,w,h,t,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.2,rounding_size=1.0',fc=fc,ec='black',lw=0.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=6.3,linespacing=1.3)
xs=[0.5,20.7,40.9,61.1,81.3]; W=17.0
top=['1. Choose a rung\n(row, 5 s, 120 s,\nblocked, chrono,\nleave-one-out)','2. Assign whole\ngroups to train,\nvalidation, test','3. Refit scaler, AE\nand classifiers\n(new seed)','4. Fix thresholds on\nvalidation benign\nscores','5. Score test set:\nMCC, recall,\nfalse positives']
bot=['6. Repeat over\nseeds\n(3 to 20 per rung)','7. Paired differences\n(bootstrap, corrected\nt-test, Bonferroni)','8. Diagnostics: ties,\nsaturation, FP\nversus target','9. Ablations: no\nscore, no ports,\nport shuffle','10. Report every\nnumber with its\nrung']
for x,t in zip(xs,top): box(x,28,W,12,t,'#CFE3F3')
for x,t in zip(xs,bot): box(x,5,W,12,t,'#FBE3B5')
for x in xs[:-1]:
    for y in (34,11): ax.annotate('',xy=(x+W+3.4,y),xytext=(x+W+0.4,y),arrowprops=dict(arrowstyle='->',lw=0.8))
ax.annotate('',xy=(xs[-1]+W/2,17.4),xytext=(xs[-1]+W/2,27.6),arrowprops=dict(arrowstyle='->',lw=0.8))
save(fig,'fig1_protocol')
# ---- Fig 2
C=[('Row','C_row'),('5 s\nrandom\nbenign','C_r5rand'),('5 s\nblocked','C_r5blk'),('120 s\nblocked','C_r120blk'),('Chrono-\nlogical','C_chrono_x')]
D=[('Row\n(blocked)','D_row_blk'),('5 s\nrandom\nbenign','D_r5rand'),('5 s\nblocked','D_r5blk'),('120 s\nrandom\nbenign','D_r120'),('Chrono-\nlogical','D_chrono')]
cols=[BLUE,GREEN,ORANGE,PURPLE,VERM]; mk=['o','s','^','D','v']
fig,axs=plt.subplots(1,2,figsize=(6.9,3.0),sharey=True)
for ax,rungs,rule,title in [(axs[0],C,'fpr4e-05','(a) CICAPT-IIoT, FPR 0.004%'),(axs[1],D,'fpr0.005','(b) DAPT2020, FPR 0.5%')]:
    for i,(lab,k) in enumerate(rungs):
        v=vals(load(k),'lgbm_full',rule); x=np.full(len(v),i)+np.linspace(-0.22,0.22,len(v))
        ax.scatter(x,v,s=14,marker=mk[i],facecolor=cols[i],edgecolor='black',lw=0.4,alpha=0.9,zorder=3)
        ax.hlines(np.median(v),i-0.34,i+0.34,color='black',lw=1.8,zorder=4); ax.text(i,-0.13,f'n={len(v)}',ha='center',fontsize=6.5)
    ax.set_xticks(range(5)); ax.set_xticklabels([r[0] for r in rungs],fontsize=7); ax.set_title(title,fontsize=8.5,loc='left')
    ax.set_ylim(-0.17,1.02); ax.grid(axis='y',lw=0.3,alpha=0.6); ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Test MCC, deployed model'); save(fig,'fig2_ladder')
# ---- Fig 3
names={'lgbm_full':'Deployed','lgbm_noscore':'No score','mlp':'MLP','lgbm_tuned':'LGBM tuned','xgb':'XGBoost','rf':'RF','et':'ET','gb':'GB','ada':'AdaBoost'}
allm=['lgbm_full','lgbm_noscore','lgbm_tuned','xgb','rf','et','gb','ada','mlp']
panels=[('(a) CICAPT-IIoT\n5 s blocked','C_r5blk','C_r5blk_x',['lgbm_full','lgbm_noscore','mlp'],'fpr4e-05'),('(b) DAPT2020, 5 s blocked','D_r5blk',None,allm,'fpr0.005'),('(c) DAPT2020, row split','D_row_blk',None,allm,'fpr0.005')]
fig,axs=plt.subplots(1,3,figsize=(6.9,3.4),gridspec_kw={'width_ratios':[3,9,9]},sharey=True)
for ax,(t,k,kx,ms,rule) in zip(axs,panels):
    R=merged(k,kx)
    for i,m in enumerate(ms):
        p,q=np.median(vals(R,m,rule)),np.median(vals(R,m+'_portshuf',rule))
        ax.plot([i,i],[p,q],color=GREY,lw=1.0,zorder=1)
        ax.scatter([i],[p],marker='o',s=28,color=BLUE,edgecolor='black',lw=0.4,zorder=3,label='ports intact' if i==0 else None)
        ax.scatter([i],[q],marker='s',s=28,color=VERM,edgecolor='black',lw=0.4,zorder=3,label='ports shuffled' if i==0 else None)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([names[m] for m in ms],fontsize=6.5,rotation=60,ha='right')
    ax.set_title(t,fontsize=8,loc='left'); ax.set_ylim(0,1.0); ax.set_xlim(-0.6,len(ms)-0.4); ax.grid(axis='y',lw=0.3,alpha=0.6); ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Median test MCC'); axs[1].legend(fontsize=6.5,frameon=False,loc='upper right'); fig.subplots_adjust(wspace=0.08)
save(fig,'fig3_portshuffle'); print('done ->',a.out)