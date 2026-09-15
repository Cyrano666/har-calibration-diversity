"""Plain comparison bars from the unchanged main-study summaries."""
from pathlib import Path
import sys,os,json
W=Path(__file__).resolve().parent;ROOT=W.parent
os.environ['MPLCONFIGDIR']=str(W/'mplcache')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf
DATA=ROOT/'revision6/summaries';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
summary=pd.read_csv(DATA/'overview.csv');failure=pd.read_csv(DATA/'failure_rates.csv')
plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9.5,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#4477AA','#EE9944','#66AA88'];ds=['har','dsads','mhealth','wisdm','pamap2','realdisp','uschad'];names=['HAR','DSADS','MHEALTH','WISDM','PAMAP2','REALDISP','USC-HAD'];pol=['horizon','hypergeom','martingale'];labels=['JRC','Bonferroni HG','Uniform-prior CS'];values=[]
def finish(fig,n):
    for ext in ['pdf','eps','png']:fig.savefig(OUT/f'Fig{n}.{ext}',bbox_inches='tight',pad_inches=.045,dpi=600)
    plt.close(fig)
def style(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E7EA',linewidth=.55);ax.tick_params(width=.7,length=3)
fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),sharey=True);fig.subplots_adjust(left=.075,right=.99,bottom=.24,top=.78,wspace=.15)
for i,(ax,score) in enumerate(zip(axes,['LAC','APS'])):
    for j,(policy,color,label) in enumerate(zip(pol,colors,labels)):
        data=summary[(summary.score==score)&(summary.tolerance==.5)&(summary.policy==policy)].set_index('dataset').loc[ds]
        y=data.saving.to_numpy()*100
        ax.bar(np.arange(7)+(j-1)*.24,y,width=.22,color=color,edgecolor='white',linewidth=.3,label=label)
        values.extend(dict(figure=2,score=score,dataset=d,policy=policy,percent=float(v)) for d,v in zip(ds,y))
    ax.set_xticks(range(7),names,rotation=35,ha='right');ax.set_ylim(0,80);ax.set_yticks([0,20,40,60,80]);style(ax)
    ax.text(0,1.055,'('+chr(97+i)+') '+score,transform=ax.transAxes,fontsize=10)
axes[0].set_ylabel('Reference labels saved (%)')
handles,lab=axes[0].get_legend_handles_labels();fig.legend(handles,lab,loc='upper center',bbox_to_anchor=(.55,1.005),ncol=3,frameon=False,handlelength=1.5,columnspacing=1.3)
finish(fig,2)
policies=['horizon','hypergeom','martingale','fixed_25','fixed_50','fixed_75','midpoint','uncorrected'];ticks=['JRC','HG','CS','25%','50%','75%','Mid','Uncorr.']
data=failure[(failure.dataset=='uschad')&(failure.score=='LAC')&(failure.tolerance==.5)].set_index('policy').loc[policies]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.25));fig.subplots_adjust(left=.075,right=.99,bottom=.24,top=.78,wspace=.25)
for i,(ax,metric,ylabel) in enumerate(zip(axes,['containment_failure','inflation_exceeded'],['Threshold failure (%)','Batch-inflation failure (%)'])):
    y=data[metric].to_numpy()*100
    ax.bar(range(8),y,width=.64,color=colors+['#AAB2BA']*5,edgecolor='white',linewidth=.3)
    threshold=ax.axhline(5,color='#AA4455',linewidth=1,linestyle=(0,(4,3)),label=r'$\delta=5\%$')
    ax.set_xticks(range(8),ticks,rotation=35,ha='right');ax.set_ylabel(ylabel);ax.set_ylim(0,50 if i==0 else 20)
    ax.set_yticks([0,10,20,30,40,50] if i==0 else [0,5,10,15,20]);style(ax)
    ax.text(0,1.055,'('+chr(97+i)+')',transform=ax.transAxes,fontsize=10)
    values.extend(dict(figure=3,metric=metric,policy=p,percent=float(v)) for p,v in zip(policies,y))
fig.legend([threshold],[r'$\delta=5\%$'],loc='upper center',bbox_to_anchor=(.55,1.005),frameon=False,handlelength=2)
finish(fig,3)
(W/'plotted_values.json').write_text(json.dumps(values,indent=2))
audit=[]
for n in [2,3]:
    d=pymupdf.open(OUT/f'Fig{n}.pdf');p=d[0];scale=174/25.4*72/p.rect.width
    spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
    minimum=min(s['size'] for s in spans)*scale;assert minimum>=8
    assert all(d.extract_font(f[0])[3] for f in p.get_fonts())
    audit.append(dict(figure=n,minimum_font_at_174mm=minimum,raster_images=len(p.get_images())))
(W/'plot_check.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
