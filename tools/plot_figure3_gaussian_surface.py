"""Integrated six-panel characterization of the Gaussian surface pathway."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
STATES = ['holo','apo','af2']
STATE_NAMES = {'holo':'Holo','apo':'Apo','af2':'AF2'}
STATE_COLORS = {'holo':'#7564A7','apo':'#C48A32','af2':'#2369A8'}
MODELS = ['gaussian_direct128','pesto_official','gpsfun_gpsite_pdb','scannet_ppi_no_msa']
MODEL_NAMES = dict(zip(MODELS,['GSBind','PeSTo','GPSFun','ScanNet']))
MODEL_COLORS = dict(zip(MODELS,['#2369A8','#D95B79','#31968C','#C99231']))
BLUE = '#2369A8'
TEAL = '#258D86'
TEXT = '#203343'


def style(axis):
    axis.spines[['top','right']].set_visible(False)
    axis.spines[['left','bottom']].set_color('#A1ACB5')
    axis.tick_params(color='#A1ACB5', labelcolor=TEXT, length=3, pad=2)
    axis.set_axisbelow(True)
    axis.grid(axis='y',color='#E8EDF1',linewidth=.5)


def heading(figure, cell, letter, title, subtitle):
    box = cell.get_position(figure)
    figure.text(box.x0-.016, box.y1+.026, letter, fontsize=15, fontweight='bold',color=TEXT)
    figure.text(box.x0+.005, box.y1+.026, title, fontsize=11, fontweight='bold',color=TEXT)
    figure.text(box.x0+.005, box.y1+.008, subtitle, fontsize=7,color='#596D7C')


def row(summary,state,condition,metric):
    return summary[(summary.input_state==state)&(summary.condition==condition)&(summary.metric==metric)].iloc[0]


def violin(axis, values, position, color, width=.46):
    values=np.asarray(values)
    if len(np.unique(values))>1:
        packet=axis.violinplot(values,positions=[position],widths=width,showextrema=False)
        for body in packet['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.16)
            body.set_linewidth(.6)
    lower,median,upper=np.quantile(values,[.25,.5,.75])
    axis.plot([position,position],[lower,upper],color=color,linewidth=2.3,zorder=4)
    axis.scatter(position,median,s=6,color='white',edgecolors=color,linewidths=.5,zorder=5)


def draw_dependency(figure,cell,data):
    axis=figure.add_subplot(cell)
    summary=data['summary']
    systems=data['systems']
    conditions=['baseline','surface_half','surface_zero']
    generator=np.random.default_rng(20261021)
    for state_index,state in enumerate(STATES):
        color=STATE_COLORS[state]
        offset=state_index*3.6
        group=systems[systems.input_state.eq(state)]
        paired=group.pivot(index='system_id',columns='condition',values='ap')
        for values in paired[conditions].to_numpy():
            axis.plot(offset+np.arange(3),values,color=color,alpha=.035,linewidth=.35,zorder=1)
        for condition_index,condition in enumerate(conditions):
            values=paired[condition].to_numpy()
            position=offset+condition_index
            violin(axis,values,position,color,.6)
            axis.scatter(position+generator.uniform(-.17,.17,len(values)),values,s=1.5,alpha=.13,color=color,rasterized=True)
        means=[row(summary,state,condition,'ap') for condition in conditions]
        axis.plot(offset+np.arange(3),[value.estimate for value in means],color=color,linewidth=1.9,zorder=6)
        axis.errorbar(offset+np.arange(3),[value.estimate for value in means],
            yerr=np.array([[value.estimate-value.lower for value in means],[value.upper-value.estimate for value in means]]),
            fmt='D',markersize=3.8,mfc='white',mec=color,color=color,capsize=2,linewidth=1.2,zorder=7)
        loss=row(summary,state,'surface_zero','ap_loss')
        axis.text(offset+1,1.045,f'{STATE_NAMES[state]}  |  {len(paired)} systems',ha='center',fontsize=7.5,color=color,fontweight='bold')
        axis.text(offset+1,.98,f'AP loss {100*loss.estimate:.1f} pp',ha='center',fontsize=7,color=color)
    axis.set_xticks([state*3.6+condition for state in range(3) for condition in range(3)],['100','50','0']*3)
    axis.set_ylim(-.025,1.09)
    axis.set_ylabel('Binding AP',color=TEXT)
    axis.set_xlabel('Gaussian surface feedback retained (%)',color=TEXT)
    style(axis)
    heading(figure,cell,'a','The surface pathway supports prediction',
        'Paired systems; distributions + IQR; diamonds: mean and 95% CI')


def draw_dose(figure,cell,data):
    axis=figure.add_subplot(cell)
    paths={'Surface feedback':['baseline','surface_half','surface_zero'],
        'Pair message':['baseline','P_half','P_zero']}
    route_colors={'Surface feedback':BLUE,'Pair message':TEAL}
    markers={'holo':'o','apo':'^','af2':'s'}
    styles={'holo':'-','apo':':','af2':'--'}
    for state in STATES:
        for route,conditions in paths.items():
            values=[row(data['summary'],state,condition,'ap_loss') for condition in conditions]
            means=np.array([value.estimate for value in values])*100
            low=np.array([value.lower for value in values])*100
            high=np.array([value.upper for value in values])*100
            axis.fill_between([0,50,100],low,high,color=route_colors[route],alpha=.055)
            axis.errorbar([0,50,100],means,yerr=[means-low,high-means],color=route_colors[route],
                marker=markers[state],linestyle=styles[state],markersize=4,linewidth=1.3,capsize=2)
    axis.axhline(0,color='#A3ADB5',linewidth=.7)
    axis.set_xticks([0,50,100])
    axis.set_xlim(-5,105)
    axis.set_ylim(-2,35)
    axis.set_xlabel('Message / feedback removed (%)')
    axis.set_ylabel('Binding AP loss (percentage points)')
    handles=[Line2D([],[],color=color,linewidth=2,label=name) for name,color in route_colors.items()]
    handles += [Line2D([],[],color='#566977',marker=markers[state],linestyle=styles[state],label=STATE_NAMES[state]) for state in STATES]
    axis.legend(handles=handles,fontsize=6.7,loc='upper left',frameon=False,ncol=2,handlelength=2,columnspacing=.9)
    style(axis)
    heading(figure,cell,'b','Dependence has a graded response',
        'Two intervention sites; three observed doses; shared system bootstrap')


def draw_targeting(figure,cell,data):
    nested=cell.subgridspec(2,2,height_ratios=[.18,1],width_ratios=[1,.18],hspace=.04,wspace=.04)
    axis=figure.add_subplot(nested[1,0])
    upper=figure.add_subplot(nested[0,0],sharex=axis)
    right=figure.add_subplot(nested[1,1],sharey=axis)
    effects=data['target'].groupby(['input_state','system_id'])[['ap_loss','correct_lost_per_budget']].mean().reset_index()
    limits=[-105,105]
    axis.axhspan(0,limits[1],xmin=(0-limits[0])/(limits[1]-limits[0]),xmax=1,color='#EAF3F8',zorder=0)
    for state in STATES:
        group=effects[effects.input_state.eq(state)]
        horizontal=100*group.ap_loss.to_numpy()
        vertical=100*group.correct_lost_per_budget.to_numpy()
        color=STATE_COLORS[state]
        axis.scatter(horizontal,vertical,color=color,s=5,alpha=.3,linewidths=0,rasterized=True)
        upper.hist(horizontal,bins=np.linspace(*limits,26),density=True,histtype='step',color=color,linewidth=.8)
        right.hist(vertical,bins=np.linspace(*limits,26),density=True,histtype='step',orientation='horizontal',color=color,linewidth=.8)
        first=data['contrasts'][(data['contrasts'].input_state==state)&data['contrasts'].metric.eq('ap_loss')].iloc[0]
        second=data['contrasts'][(data['contrasts'].input_state==state)&data['contrasts'].metric.eq('correct_lost_per_budget')].iloc[0]
        axis.errorbar(100*first.estimate,100*second.estimate,
            xerr=[[100*(first.estimate-first.lower)],[100*(first.upper-first.estimate)]],
            yerr=[[100*(second.estimate-second.lower)],[100*(second.upper-second.estimate)]],
            fmt='D',mfc='white',mec=color,color=color,markersize=5,capsize=2,linewidth=1.2,zorder=8)
    axis.axhline(0,color='#899AA7',linewidth=.7)
    axis.axvline(0,color='#899AA7',linewidth=.7)
    axis.set(xlim=limits,ylim=limits,xlabel='Extra AP loss: targeted minus random (pp)',
        ylabel='Extra correct top-10% candidates lost (pp)')
    axis.legend(handles=[Line2D([],[],color=STATE_COLORS[state],marker='o',linestyle='',label=STATE_NAMES[state]) for state in STATES],
        frameon=False,fontsize=6.5,loc='lower right')
    inset=axis.inset_axes([.055,.64,.39,.27])
    for state_index,state in enumerate(STATES):
        for metric_index,metric in enumerate(['interface_priority_loss','background_priority_loss']):
            current=data['contrasts'][(data['contrasts'].input_state==state)&data['contrasts'].metric.eq(metric)].iloc[0]
            position=metric_index+(state_index-1)*.2
            inset.errorbar(position,100*current.estimate,
                yerr=[[100*(current.estimate-current.lower)],[100*(current.upper-current.estimate)]],
                fmt='o',markersize=2,color=STATE_COLORS[state],capsize=1,linewidth=.7)
    inset.axhline(0,color='#A1ACB5',lw=.6)
    inset.set_xticks([0,1],['Interface','Background'],fontsize=5)
    inset.set_title('Extra rank-priority loss (pp)',fontsize=5.5,loc='left',pad=2)
    inset.tick_params(axis='y',labelsize=5,length=2,pad=1)
    inset.spines[['top','right']].set_visible(False)
    inset.set_facecolor('#FFFFFF')
    style(axis)
    for marginal in [upper,right]:
        marginal.axis('off')
    heading(figure,cell,'c','Local pair removal disrupts priority',
        'High-score regions vs same-count random surface points; three seeds')


def draw_transfer(figure,cell,data):
    axis=figure.add_subplot(cell)
    cases=data['pairs']
    systems=cases.groupby(['input_state','system_id','model'])[['holo_ap','state_ap','ap_change']].mean().reset_index()
    generator=np.random.default_rng(20261021)
    positions=[]
    labels=[]
    for state_index,state in enumerate(['apo','af2']):
        for model_index,model in enumerate(MODELS):
            position=state_index*4.7+model_index
            group=systems[systems.input_state.eq(state)&systems.model.eq(model)]
            color=MODEL_COLORS[model]
            violin(axis,group.state_ap,position+.12,color,.58)
            jitter=generator.uniform(-.1,.1,len(group))
            for shift,reference,current in zip(jitter,group.holo_ap,group.state_ap):
                axis.plot([position-.18+shift,position+.18+shift],[reference,current],color=color,alpha=.045,linewidth=.4)
            axis.scatter(position+.18+jitter,group.state_ap,color=color,s=1.4,alpha=.18,rasterized=True)
            reference=float(group.holo_ap.mean())
            current=float(group.state_ap.mean())
            axis.annotate('',xy=(position+.18,current),xytext=(position-.18,reference),
                arrowprops=dict(arrowstyle='->',color=color,lw=1.3))
            axis.scatter(position-.18,reference,facecolors='white',edgecolors=color,s=18,zorder=7)
            summary=data['pair_summary'][(data['pair_summary'].input_state==state)&data['pair_summary'].model.eq(model)&data['pair_summary'].metric.eq('state_ap')].iloc[0]
            axis.errorbar(position+.18,current,yerr=[[current-summary.lower],[summary.upper-current]],fmt='D',color=color,mfc=color,ms=3,capsize=2,zorder=7)
            positions.append(position)
            labels.append(MODEL_NAMES[model].replace('GSBind','GS'))
        subset=cases[cases.input_state.eq(state)]
        axis.text(state_index*4.7+1.5,1.08,f'{STATE_NAMES[state]} | {subset.case_id.nunique()} cases / {subset.system_id.nunique()} systems',
            ha='center',fontsize=7.5,fontweight='bold',color=TEXT)
    axis.set_xticks(positions,labels,rotation=25,ha='right')
    axis.set_ylim(-.025,1.12)
    axis.set_ylabel('Binding AP on the same aligned reference interface')
    axis.set_xlabel('Model  |  open: holo reference; filled: unbound input')
    axis.axvline(3.85,color='#DDE5EB',linewidth=.8)
    style(axis)
    heading(figure,cell,'d','Accurate interfaces survive input changes',
        'One line per system; common positions and fixed holo labels')


def draw_correct_stability(figure,cell,data):
    nested=cell.subgridspec(1,2,wspace=.08)
    axes=[figure.add_subplot(nested[0,index]) for index in range(2)]
    values=data['stability']
    values=values[values.requested_k.eq(0)].groupby(['input_state','system_id','model'])[['jaccard','retained_correct_per_budget']].mean().reset_index()
    for axis,state in zip(axes,['apo','af2']):
        overlap=np.linspace(0,1,200)
        axis.fill_between(overlap,2*overlap/(1+overlap),1,color='#F2F4F6',zorder=0)
        axis.plot(overlap,2*overlap/(1+overlap),color='#A6B2BC',linewidth=.8,linestyle='--')
        for model in MODELS:
            group=values[values.input_state.eq(state)&values.model.eq(model)]
            color=MODEL_COLORS[model]
            axis.scatter(group.jaccard,group.retained_correct_per_budget,color=color,s=4,alpha=.16,linewidths=0,rasterized=True)
            summaries=data['stability_summary']
            horizontal=summaries[(summaries.input_state==state)&summaries.model.eq(model)&summaries.requested_k.eq(0)&summaries.metric.eq('jaccard')].iloc[0]
            vertical=summaries[(summaries.input_state==state)&summaries.model.eq(model)&summaries.requested_k.eq(0)&summaries.metric.eq('retained_correct_per_budget')].iloc[0]
            axis.errorbar(horizontal.estimate,vertical.estimate,
                xerr=[[horizontal.estimate-horizontal.lower],[horizontal.upper-horizontal.estimate]],
                yerr=[[vertical.estimate-vertical.lower],[vertical.upper-vertical.estimate]],fmt='D',mfc='white',mec=color,color=color,ms=4,capsize=2,linewidth=1,zorder=8)
        axis.set(xlim=(0,1),ylim=(0,1),xlabel='Candidate overlap (Jaccard)')
        axis.text(.93,.97,STATE_NAMES[state],fontsize=8,fontweight='bold',color=TEXT,transform=axis.transAxes,ha='right',va='top')
        axis.set_xticks([0,.5,1])
        axis.set_yticks([0,.25,.5,.75,1])
        style(axis)
    axes[0].set_ylabel('Shared true interface hits / candidate budget')
    axes[1].tick_params(labelleft=False)
    axes[0].text(.03,.97,'Dashed: all shared\ncandidates correct',transform=axes[0].transAxes,fontsize=6,color='#6C7D89',va='top')
    axes[1].legend(handles=[Line2D([],[],color=MODEL_COLORS[model],marker='o',linestyle='',label=MODEL_NAMES[model]) for model in MODELS],
        frameon=False,fontsize=6,loc='lower right',handletextpad=.3)
    heading(figure,cell,'e','Preserve correct candidates, not just overlap',
        'Fixed top-10% budgets; system clouds + paired mean uncertainty')


def draw_structure(figure,cell,data,folder):
    nested=cell.subgridspec(2,3,height_ratios=[1,.68],hspace=.15,wspace=.03)
    table=pd.read_csv(folder/'F_example_residues.csv')
    example=pd.read_csv(folder/'F_example_metrics.csv').set_index('condition')
    selection=json.loads((folder/'F_EXAMPLE_SELECTION.json').read_text())
    with np.load(folder/'F_example_geometry.npz') as packet:
        centers=packet['centers']
        surface=packet['surface_positions']
        local_mask=packet['P_top']
    center=surface.mean(axis=0)
    _,_,basis=np.linalg.svd(surface-center,full_matrices=False)
    surface_view=(surface-center)@basis.T
    residue_view=(centers-center)@basis.T
    nearest=cKDTree(centers).query(surface)[1]
    conditions=['baseline','P_top','surface_zero']
    names=['Native','Local pair off','Surface off']
    normal=Normalize(0,1)
    image=None
    labels=table.label.to_numpy(int)
    valid=table.label_mask.astype(bool).to_numpy()
    shared_limits=np.max(np.abs(surface_view[:,:2]),axis=0)*1.07
    depth_order=np.argsort(surface_view[:,2])
    for index,(condition,name) in enumerate(zip(conditions,names)):
        axis=figure.add_subplot(nested[0,index])
        image=axis.scatter(surface_view[depth_order,0],surface_view[depth_order,1],
            c=table[condition].to_numpy()[nearest[depth_order]],s=5,cmap='coolwarm',norm=normal,linewidths=0,rasterized=True)
        positive=valid&(labels==1)
        axis.scatter(residue_view[positive,0],residue_view[positive,1],s=8,facecolors='none',edgecolors='#D69E24',linewidths=.55,zorder=4)
        if condition=='P_top':
            axis.scatter(surface_view[local_mask,0],surface_view[local_mask,1],s=6,facecolors='none',edgecolors='#253B4C',linewidths=.17,alpha=.4,zorder=3)
        score=table[condition].to_numpy()[valid]
        chosen=np.argsort(-score,kind='stable')[:int(np.ceil(.1*valid.sum()))]
        hits=int(labels[valid][chosen].sum())
        axis.set_title(f'{name}\nAP {example.loc[condition,"ap"]:.3f} | hits {hits}/{len(chosen)}',fontsize=7,color=TEXT,pad=1)
        axis.set(xlim=(-shared_limits[0],shared_limits[0]),ylim=(-shared_limits[1],shared_limits[1]))
        axis.set_aspect('equal')
        axis.axis('off')
    axis=figure.add_subplot(nested[1,:])
    order=np.argsort(-table.loc[valid,'baseline'].to_numpy(),kind='stable')
    chosen_table=table[valid].iloc[order]
    rank=np.arange(len(chosen_table))/len(chosen_table)*100
    mask=chosen_table.label.eq(1).to_numpy()
    for condition,color,name in zip(['P_top','surface_zero'],[TEAL,BLUE],['Local pair off','Surface off']):
        shift=(chosen_table['baseline_rank']-chosen_table[condition+'_rank']).to_numpy()*100
        axis.scatter(rank[mask],shift[mask],color=color,s=5,alpha=.7,label=name,linewidths=0)
    axis.axhline(0,color='#8B9CA9',lw=.7)
    axis.axvspan(0,10,color='#E8EFF5',zorder=0)
    axis.set(xlim=(0,100),xlabel='Native score rank (% of valid residues)',ylabel='Interface priority\nlost (percentile points)')
    axis.tick_params(labelsize=6)
    axis.legend(fontsize=5.8,frameon=False,loc='upper right',ncol=2,handletextpad=.2,columnspacing=.6)
    style(axis)
    box=cell.get_position(figure)
    color_axis=figure.add_axes([box.x1-.006,box.y0+.17,.004,.12])
    bar=figure.colorbar(image,cax=color_axis,ticks=[0,.5,1])
    bar.ax.tick_params(labelsize=5,length=1)
    bar.set_label('Binding score',fontsize=5,labelpad=1)
    pdb=selection['system_id'].split('__')[0]
    side=selection['case_id'].split('::')[-1]
    heading(figure,cell,'f','Gaussian support localizes the interface',
        f'{pdb} {side} | gold: true interface; surface color: nearest-residue score')


def run(folder,figure_dir=None):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
    figures=figure_dir if figure_dir is not None else folder/'figures'
    figures.mkdir(exist_ok=True)
    names={'summary':'intervention_summary.csv','systems':'intervention_system_metrics.csv',
        'target':'target_minus_random_case_metrics.csv','contrasts':'target_minus_random_summary.csv',
        'pairs':'pair_model_metrics.csv','pair_summary':'A_pair_and_H_packing_summary.csv',
        'stability':'stability_case_metrics.csv','stability_summary':'C_stability_summary.csv'}
    index_path=folder/'DATA_FILE_INDEX.csv'
    index=pd.read_csv(index_path).set_index('file').relative_path.to_dict() if index_path.exists() else {}
    data={key:pd.read_csv(folder/index.get(name,name)) for key,name in names.items()}
    example_folder=folder/'F_Spatial_example' if (folder/'F_Spatial_example').exists() else folder
    figure=plt.figure(figsize=(18,11.5),facecolor='white')
    layout=figure.add_gridspec(2,3,left=.055,right=.98,bottom=.09,top=.88,wspace=.29,hspace=.35)
    figure.text(.055,.963,'Gaussian surface support for accurate binding-site prediction',
        fontsize=19,fontweight='bold',color=TEXT)
    figure.text(.055,.936,'A frozen model, controlled pathway interventions, and sequence-audited conformational pairs',
        fontsize=10,color='#617585')
    figure.text(.055,.918,'COMPUTATIONAL DEPENDENCE  /  DOSE RESPONSE  /  SITE PRIORITIZATION',fontsize=8,fontweight='bold',color=BLUE)
    draw_dependency(figure,layout[0,0],data)
    draw_dose(figure,layout[0,1],data)
    draw_targeting(figure,layout[0,2],data)
    draw_transfer(figure,layout[1,0],data)
    draw_correct_stability(figure,layout[1,1],data)
    draw_structure(figure,layout[1,2],data,example_folder)
    figure.text(.055,.035,'Interventions: 988 holo / 84 apo / 790 AF2 cases. Transfer: 70 apo / 718 AF2 uniquely aligned cases. '
        'Means are system-equal; 95% CIs use 10,000 system resamples.',fontsize=7.5,color='#617585')
    for suffix in ['pdf','png']:
        figure.savefig(figures/f'GSBind_figure3_gaussian_surface_v20261021.{suffix}',dpi=260,facecolor='white')
    figure.canvas.draw()
    for letter,cell in zip('ABCDEF',[layout[0,0],layout[0,1],layout[0,2],layout[1,0],layout[1,1],layout[1,2]]):
        box=cell.get_position(figure)
        bounds=matplotlib.transforms.Bbox.from_extents((box.x0-.025)*18,(box.y0-.07)*11.5,
            min(1,box.x1+.025)*18,(box.y1+.055)*11.5)
        for suffix in ['pdf','png']:
            figure.savefig(figures/f'{letter}_gaussian_surface.{suffix}',bbox_inches=bounds,dpi=260,facecolor='white')
    plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'results/figure3_gaussian_surface_20261021')
    parser.add_argument('--figure-dir',type=Path)
    arguments=parser.parse_args()
    run(arguments.data_dir,arguments.figure_dir)
