"""Plots and source-data export for surrogate training and evaluation."""
from pathlib import Path

import numpy as np

from ..plotting import setup, save


def plot_training(out,history):
    plt=setup();fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    keys=('train_quality_d50_um','train_quality_span','validation_score','validation_normalized_regret')
    labels=('Train D50 normalized MSE','Train Span normalized MSE','Fixed development quality score','Terminal candidate ranking penalty (eligibility determined by PBM truth)')
    for ax,key,label in zip(axes.flat,keys,labels):
        rs=[r for r in history if r.get(key) is not None]
        ax.plot([r['epoch'] for r in rs],[r[key] for r in rs],color='#3676B5')
        ax.set(xlabel='Planned epoch',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle(Path(out).name+': control-oriented training; closed loop not yet validated')
    save(plt,fig,Path(out)/'figures','control_training')
    (Path(out)/'figures'/'figure_notes.txt').write_text('Source data: ../history.csv.\nTraining quality curves show normalized MSE; validation sets and scores are fixed.\n'
        'Terminal rankings include penalties for predicted ties or no feasible prediction; PBM ties are ineligible. Missing points are unmeasured/invalid, not zero error.\n',encoding='utf-8')


def plot_evaluation(out,records,responses,rankings,show=False):
    plt=setup(show);out=Path(out);figures=out/'figures'
    for family in sorted({r['family'] for r in records}):
        parent=sorted({r['parent_id'] for r in records if r['family']==family})[0]
        fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
        for ax,key,label in zip(axes.flat,('d50_error_um','span_error','yield_error_pp','mass_l1'),
                               ('D50 error (um)','Span error','Yield error (percentage points)','Mass-CSD relative L1 (auxiliary)')):
            for kind,origin,color,legend in [('one_step',-1,'#3676B5','True-history one step'),('recursive',10,'#D16A43','Recursive prediction from minute 10')]:
                rs=[r for r in records if r['parent_id']==parent and r['kind']==kind and r['origin_min']==origin]
                ax.plot([r['endpoint_min'] for r in rs],[r.get(key,np.nan) if r['valid'] else np.nan for r in rs],color=color,label=legend)
            ax.axhline(0,color='.6',lw=.6);ax.grid(alpha=.2);ax.set(xlabel='Prediction endpoint (min)',ylabel=label)
        axes.flat[0].legend();fig.suptitle(parent+': known actual temperature; no future state feedback')
        save(plt,fig,figures,parent+'_control_prediction',show)
    if rankings:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        valid=[r for r in rankings if r.get('truth_cost_of_predicted_choice') is not None and r.get('best_truth_objective') is not None]
        if valid:
            x=[r['best_truth_objective'] for r in valid];y=[r['truth_cost_of_predicted_choice'] for r in valid]
            ax.scatter(x,y,s=18,color='#3676B5');lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],'--',color='.5')
        ax.set(xlabel='Lowest PBM cost among evaluated candidates',ylabel='PBM cost of the surrogate-selected candidate');ax.grid(alpha=.2)
        fig.suptitle('Finite terminal candidate selection; not a global optimality proof')
        save(plt,fig,figures,'terminal_candidate_selection',show)
    figures.mkdir(exist_ok=True)
    (figures/'figure_notes.txt').write_text('Source data: ../per_minute.csv, response.csv, terminal_ranking.csv, candidate_scores.csv.\n'
        'The first fixed ID per family is plotted; CSV retains all selected parents. Invalid segments remain gaps. Actual T is known, excluding execution uncertainty.\n'
        'Terminal scatter shows selections that are also PBM-valid and feasible. Misclassifications and uninformative groups remain in CSV and summaries.\n'
        'Candidates in a group and minutes from one trajectory are correlated, not independent samples. No new PBM or control optimization was run.\n',encoding='utf-8')
    if show:plt.show()
