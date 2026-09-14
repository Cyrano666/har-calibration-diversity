"""Revised journal-sized figures; frozen V6 quantitative inputs are unchanged."""
from pathlib import Path
import sys,os,json,shutil
W=Path(__file__).resolve().parent;V6=W.parent/'revision6';BASE=W.parent/('v2' if (W.parent/'v2').exists() else 'revision')
sys.path[:0]=[str(W.parent/'revision5/fitdeps'),str(W.parent/'revision4/plotdeps'),str(V6),str(BASE),str(W.parent/'revision4/pydeps')]
os.environ['MPLCONFIGDIR']=str(W/'mplcache')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from designs import scores,cutoff
from sequential import trace,stop
import pymupdf
OUT=W/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':9.5,'legend.fontsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.5})
o=pd.read_csv(V6/'summaries/overview.csv');fail=pd.read_csv(V6/'summaries/failure_rates.csv');ci=pd.read_csv(V6/'summaries/paired_intervals.csv');models=pd.read_csv(V6/'summaries/model_summary.csv')
DS=['har','dsads','mhealth','wisdm','pamap2','realdisp','uschad'];names=['HAR','DSADS','MHEALTH','WISDM','PAMAP2','REALDISP','USC-HAD'];col=['#185A85','#B46B13','#167060'];pols=['horizon','hypergeom','martingale'];labels=['JRC','Bonferroni HG','Prior/posterior CS'];markers=['o','s','^','D','v','P','X'];palette=['#17618C','#BD6A13','#247D45','#9F3237','#8157A3','#765642','#B4438A']
def panel(ax,i):ax.text(-.10,1.03,'('+chr(97+i)+')',transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
def save(fig,n):
 fig.tight_layout(pad=.9)
 for ext in ['pdf','eps','png']:fig.savefig(OUT/f'Fig{n}.{ext}',bbox_inches='tight',pad_inches=.03,dpi=600)
 plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.5,3.1),sharey=True)
for i,(ax,score) in enumerate(zip(axs,['LAC','APS'])):
 for j,(pol,color,label) in enumerate(zip(pols,col,labels)):
  a=o[(o.score==score)&(o.tolerance==.5)&(o.policy==pol)].set_index('dataset').loc[DS]
  ax.bar(np.arange(7)+(j-1)*.25,a.saving*100,.24,color=color,hatch=['','//','xx'][j],edgecolor='black',lw=.45,label=label)
 ax.set_xticks(range(7),names,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_axisbelow(True);ax.grid(axis='y',color='#e0e0e0',lw=.6);panel(ax,i)
axs[0].set_ylabel('Labels saved (%)');axs[1].legend(loc='upper right',frameon=False);save(fig,2)

fig,axs=plt.subplots(1,2,figsize=(7.5,3.0));pol=['horizon','hypergeom','martingale','fixed_25','fixed_50','fixed_75','midpoint','uncorrected'];ticks=['JRC','HG','CS','25%','50%','75%','Mid','Uncorr.']
a=fail[(fail.dataset=='uschad')&(fail.score=='LAC')&(fail.tolerance==.5)].set_index('policy').loc[pol]
for i,(ax,metric,ylabel) in enumerate(zip(axs,['containment_failure','inflation_exceeded'],['Threshold violations (%)','Size-inflation violations (%)'])):
 ax.bar(range(8),a[metric]*100,color=[col[0]]+['#8C9BA6']*7,edgecolor='black',lw=.5)
 ax.axhline(5,color='#943331',ls='--',lw=1.2);ax.set_xticks(range(8),ticks,rotation=35,ha='right');ax.set_ylabel(ylabel);panel(ax,i)
save(fig,3)

fig,axs=plt.subplots(1,2,figsize=(7.5,3.1),sharey=True)
for i,(ax,base,color) in enumerate(zip(axs,['hypergeom','martingale'],col[:2])):
 a=ci[(ci.score=='LAC')&(ci.baseline==base)&(ci.metric=='saving')].set_index('dataset').loc[DS]
 ax.errorbar(a.difference*100,np.arange(7),xerr=np.vstack([(a.difference-a.ci_low)*100,(a.ci_high-a.difference)*100]),fmt='o',color=color,capsize=3,ms=5)
 ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(7),names);ax.set_xlabel('Additional savings (percentage points)');ax.grid(axis='x',color='#e0e0e0',lw=.6);panel(ax,i)
axs[0].invert_yaxis();save(fig,4)

fig,axs=plt.subplots(1,2,figsize=(7.5,3.2),sharey=True)
for j,(ax,score) in enumerate(zip(axs,['LAC','APS'])):
 for i,ds in enumerate(DS):
  a=o[(o.score==score)&(o.policy=='horizon')&(o.dataset==ds)].sort_values('tolerance');ax.plot(a.tolerance,a.saving*100,marker=markers[i],ms=4.5,color=palette[i],ls=['-','--',':','-.','-','--',':'][i],label=names[i])
 ax.set_xlabel('Allowed mean extra labels');ax.set_xticks([.1,.25,.5,1]);ax.set_ylim(0,100);ax.grid(color='#e0e0e0',lw=.6);panel(ax,j)
axs[0].set_ylabel('Labels saved (%)');axs[1].legend(frameon=False,ncol=2,loc='upper left',columnspacing=.7,handlelength=1.5);save(fig,5)

z=np.load(V6/'results/predictions/uschad_0_0_MR.npz');tm=scores(z['test_p'],'LAC');cm=scores(z['cal_p'],'LAC');truth=cm[np.arange(len(cm)),z['cal_y']];rng=np.random.default_rng(617290);order=rng.permutation(np.unique(z['cal_g']));ix=rng.permutation(np.concatenate([rng.permutation(np.flatnonzero(z['cal_g']==u))[:120] for u in order]));s=truth[ix];q=cutoff(s,.1);times=tuple(range(20,481,20))
fig,axs=plt.subplots(1,2,figsize=(7.5,3.1));display_floor=1e-16
for i,(engine,color,label) in enumerate(zip(pols,col,labels)):
 records=trace(s,tm,times=times,engine=engine);chosen=stop(records,.5);t=[r['t'] for r in records];prob=[max(display_floor,1-r['upper']) for r in records]
 axs[0].plot(t,prob,color=color,ls=['-','--','-.'][i],label=label)
 axs[1].plot(t,[r['set_width'] for r in records],color=color,ls=['-','--','-.'][i]);axs[1].scatter([chosen['t']],[chosen['set_width']],marker=markers[i],color=color,s=38,zorder=4)
axs[0].set_yscale('log');axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:'1e'+str(int(np.round(np.log10(x)))) if x>0 else '0'));axs[0].axhline(max(display_floor,1-q),color='black',ls=':',lw=1,label='Full reference');axs[0].set_ylabel('Probability cutoff (1 - upper bound)');axs[0].legend(frameon=False,loc='lower right')
axs[1].axhline(.5,color='black',ls=':',lw=1);axs[1].set_ylabel('Endpoint set-size difference');axs[1].set_ylim(0,5)
for i,ax in enumerate(axs):ax.set_xlabel('Queried labels');ax.grid(color='#e0e0e0',lw=.6);panel(ax,i)
save(fig,6)
(W/'figure6_trace.json').write_text(json.dumps(dict(case='uschad_0_0_MR',draw=0,N=480,reference_q=float(q),probability_cutoff=float(1-q),plot_floor=display_floor,changed_algorithm=False),indent=2))

fig,axs=plt.subplots(1,2,figsize=(7.5,3.1))
for i,ds in enumerate(DS):
 for j,model in enumerate(['LR','MR','IT']):
  a=models[(models.dataset==ds)&(models.model==model)&(models.score=='LAC')&(models.policy=='horizon')&(models.tolerance==.5)].iloc[0];r=models[(models.dataset==ds)&(models.model==model)&(models.score=='LAC')&(models.policy=='full_reference')].iloc[0]
  axs[0].scatter(r.set_size,a.saving*100,color=palette[i],marker=markers[i],s=36,label=names[i] if j==0 else None)
  axs[1].scatter(r.coverage*100,a.coverage*100,color=palette[i],marker=markers[i],s=36)
axs[0].set_xlabel('Full-reference mean set size');axs[0].set_ylabel('Labels saved (%)');axs[0].legend(frameon=False,ncol=2,loc='upper right',columnspacing=.7,handletextpad=.3)
low=5*np.floor(min(np.min(c.get_offsets()) for c in axs[1].collections)/5)
axs[1].plot([low,100],[low,100],':',color='gray',lw=1);axs[1].set_xlim(low,100);axs[1].set_ylim(low,100);axs[1].set_xlabel('Full-reference coverage (%)');axs[1].set_ylabel('Returned-set coverage (%)')
for i,ax in enumerate(axs):panel(ax,i)
save(fig,7)
if not (OUT/'Fig1.pdf').exists():
 source=next(p for p in [V6/'Fig1.pdf',W.parent/'figures/Fig1.pdf'] if p.exists())
 shutil.copy2(source,OUT/'Fig1.pdf')
audit=[]
for p in sorted(OUT.glob('Fig*.pdf')):
 d=pymupdf.open(p);factor=174/25.4*72/d[0].rect.width;spans=[s for b in d[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
 fonts={f[0] for f in d[0].get_fonts()};minfont=min(s['size'] for s in spans)*factor
 audit.append(dict(figure=p.name,minimum_final_pt=minfont,all_fonts_embedded=all(d.extract_font(x)[3] for x in fonts),raster_images=len(d[0].get_images())))
(W/'figure_audit_after.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
