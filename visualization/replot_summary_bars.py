"""Replot frozen summaries without changing any estimates."""
from pathlib import Path
import sys,os,json,shutil
W=Path(__file__).resolve().parent;ROOT=W.parent
os.environ['MPLCONFIGDIR']=str(W/'mplcache')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf
BASE=ROOT/'revision6/summaries';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
o=pd.read_csv(BASE/'overview.csv');fail=pd.read_csv(BASE/'failure_rates.csv')
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,'legend.fontsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'hatch.linewidth':.45})
ds=['har','dsads','mhealth','wisdm','pamap2','realdisp','uschad'];names=['UCI HAR','DSADS','MHEALTH','WISDM','PAMAP2','REALDISP','USC-HAD'];colors=['#185A85','#A86413','#167060'];policies=['horizon','hypergeom','martingale'];labels=['JRC','Bonferroni HG','Uniform-prior CS']
def save(fig,n):
    for ext in ['pdf','eps','png']:fig.savefig(OUT/f'Fig{n}.{ext}',bbox_inches='tight',pad_inches=.04,dpi=600)
    plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.3,4.65),sharey=True)
fig.subplots_adjust(left=.13,right=.985,bottom=.13,top=.875,wspace=.15)
data=[]
for i,(ax,score) in enumerate(zip(axs,['LAC','APS'])):
    for j,(p,c,label) in enumerate(zip(policies,colors,labels)):
        a=o[(o.score==score)&(o.tolerance==.5)&(o.policy==p)].set_index('dataset').loc[ds]
        values=a.saving.to_numpy()*100;y=np.arange(7)+(j-1)*.24
        ax.barh(y,values,height=.21,color=c,edgecolor='#202020',lw=.45,hatch=['','//','xx'][j],label=label)
        for yy,val,d in zip(y,values,ds):
            ax.text(val+1.1,yy,f'{val:.1f}',va='center',ha='left',fontsize=9.5,color='#202020')
            data.append(dict(figure=2,score=score,dataset=d,policy=p,percent=float(val)))
    ax.set_yticks(range(7),names);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Reference labels saved (%)')
    ax.set_axisbelow(True);ax.grid(axis='x',color='#E2E5E8',lw=.6)
    ax.text(0,1.035,'('+chr(97+i)+') '+score,transform=ax.transAxes,fontweight='bold',fontsize=10)
axs[0].invert_yaxis()
handles,legend=axs[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.56,1.0),ncol=3,frameon=False,handlelength=1.6,columnspacing=1.1)
save(fig,2)
pol=['horizon','hypergeom','martingale','fixed_25','fixed_50','fixed_75','midpoint','uncorrected'];ticks=['JRC','HG','CS','25%','50%','75%','Mid','Uncorr.']
a=fail[(fail.dataset=='uschad')&(fail.score=='LAC')&(fail.tolerance==.5)].set_index('policy').loc[pol]
fig,axs=plt.subplots(1,2,figsize=(7.3,3.25));fig.subplots_adjust(left=.095,right=.985,top=.90,bottom=.21,wspace=.23)
for i,(ax,col,ylabel) in enumerate(zip(axs,['containment_failure','inflation_exceeded'],['Threshold failure (%)','Batch-inflation failure (%)'])):
    values=a[col].to_numpy()*100
    bars=ax.bar(range(8),values,color=colors+['#A8B1BA']*5,edgecolor='#303030',lw=.45)
    for j,b in enumerate(bars):b.set_hatch(['','//','xx'][j] if j<3 else '..')
    for x,v in enumerate(values):ax.text(x,v+.45,f'{v:.2f}' if v<1 else f'{v:.1f}',ha='center',va='bottom',fontsize=9,color='#202020')
    ax.axhline(5,color='#97352D',ls='--',lw=1,label=r'$\delta=5\%$');ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax.set_yticks([0,10,20,30,40,50] if i==0 else [0,5,10,15])
    ax.set_xticks(range(8),ticks,rotation=35,ha='right');ax.set_ylim(0,50 if i==0 else 18);ax.set_ylabel(ylabel)
    ax.text(-.10,1.035,'('+chr(97+i)+')',transform=ax.transAxes,fontweight='bold',fontsize=10)
save(fig,3)
(W/'plotted_values.json').write_text(json.dumps(data,indent=2))
audit=[]
for n in [2,3]:
    p=OUT/f'Fig{n}.pdf';d=pymupdf.open(p);scale=174/25.4*72/d[0].rect.width
    spans=[s for b in d[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
    audit.append(dict(figure=n,minimum_font_at_174mm=min(s['size'] for s in spans)*scale,all_fonts_embedded=all(d.extract_font(f[0])[3] for f in d[0].get_fonts()),raster_images=len(d[0].get_images())))
assert all(x['minimum_font_at_174mm']>=8 for x in audit),audit
(W/'plot_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
