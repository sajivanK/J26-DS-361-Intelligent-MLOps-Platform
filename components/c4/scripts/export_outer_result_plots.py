"""Export figures from saved final reports; never modify decisions or evaluate models."""
from pathlib import Path
import numpy as np
from _common import (parser,resolve_data_root,checked_json,checked_bytes,require,sha,
                     save_json,utc_now,git_context)
from _outer_evaluation_inputs import VERSION,REPORT_VERSION,SNAPSHOT_HASH,sources


def render_figures(governance,forecast,interval,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    methods=list(governance);positions=np.arange(len(methods));figures={}
    fig,ax=plt.subplots(figsize=(9,8.5),layout='constrained')
    left=np.zeros(len(methods))
    for name,label,color in [('auto','AUTO_DEPLOY','#25805f'),('review','HUMAN_REVIEW','#ca9a31'),('escalate','ESCALATE','#b85353')]:
        values=np.array([governance[m]['counts'][name]/governance[m]['counts']['n'] for m in methods])
        ax.barh(positions,values,left=left,label=label,color=color);left+=values
    ax.set_yticks(positions,methods);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Share of admitted outer scenarios (56 cases)')
    ax.set_title('Frozen outer decisions — Adult simulation pilot');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncols=3)
    figures['outer_decision_distribution']=fig
    fig,axes=plt.subplots(1,3,figsize=(14,8.5),layout='constrained',sharey=True)
    for ax,metric,label in zip(axes,('equal_group_automation','equal_group_unsafe_auto_frequency','human_review_burden'),
                               ('Automation','Unsafe AUTO frequency','Human review burden')):
        values=[governance[m][metric] for m in methods]
        ax.barh(positions,values,color='#42698b');ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_title(label)
        ax.set_yticks(positions,methods)
    axes[0].invert_yaxis();fig.suptitle('Equal weight per nonempty outer group (14 groups)')
    figures['outer_group_governance']=fig
    fig,ax=plt.subplots(figsize=(9,8.5),layout='constrained')
    values=[governance[m]['equal_group_conditional_unsafe_rate'] for m in methods]
    for n,value in enumerate(values):
        if value is None:ax.text(.02,n,'Undefined: no AUTO_DEPLOY',va='center',color='#666666')
        else:ax.barh(n,value,color='#b85353')
    ax.set_yticks(positions,methods);ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Equal-group unsafe fraction among deployed cases');ax.set_title('Conditional unsafe risk — undefined values are not zero')
    figures['outer_conditional_risk']=fig
    targets=('future_global_margin','future_slice_margin','future_nonaffected_margin')
    labels=('Global','True injected slice','True slice complement');models=list(forecast)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');width=.8/len(models);x=np.arange(3)
    for n,model in enumerate(models):
        values=[100*forecast[model][t]['equal_group_weighted']['mae'] for t in targets]
        ax.bar(x-.4+width/2+n*width,values,width,label=model)
    ax.set_xticks(x,labels);ax.set_ylabel('Median-forecast MAE (accuracy percentage points)')
    ax.set_title('Held-out forecast errors — equal group weighting');ax.legend(ncols=3)
    figures['outer_forecast_mae']=fig
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    comparable=[m for m in interval if interval[m]['three_target_comparable']]
    values=[interval[m]['group_all_variant_simultaneous_coverage'] for m in comparable]
    ax.barh(range(len(comparable)),values,color='#42698b')
    ax.set_yticks(range(len(comparable)),[m+(' (full bounds)' if interval[m]['fallback'] else '') for m in comparable])
    ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.axvline(.95,color='#b85353',linestyle='--',label='Nominal 95% reference')
    ax.set_xlabel('Fraction of nonempty groups with every variant and all three targets covered')
    ax.set_title('Outer group joint coverage — A4 excluded (two targets)');ax.legend()
    figures['outer_joint_coverage']=fig
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');shown=0
    for method in ('PROPOSED','B1','B2','B3','B4'):
        curve=curves[method]['curve']
        if len(curve)>=2:
            ax.step([r['equal_group_coverage'] for r in curve],[r['conditional_unsafe_rate'] for r in curve],where='post',label=method)
            shown+=1
    if shown:ax.legend()
    else:ax.text(.5,.5,'Insufficient distinct score blocks',ha='center',transform=ax.transAxes)
    ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xlabel('Forced score-prefix group coverage');ax.set_ylabel('Conditional unsafe fraction')
    ax.set_title('Diagnostic risk–coverage curves — not operational deployment policies')
    figures['outer_diagnostic_risk_coverage']=fig
    return figures


def main():
    p=parser(__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    root=resolve_data_root(args.data_root)
    directory=root/'c4_artifacts/evaluations/adult'/VERSION/REPORT_VERSION
    metadata,digest=checked_json(directory)
    require(metadata['purpose']=='verified_final_outer_evaluation' and metadata['outer_test_evaluated'] is True and
            metadata['binding']['recovered_snapshot_metadata_sha256']==SNAPSHOT_HASH and
            metadata['binding']['source_sha256']==sources(), 'Saved outer report/version/code differs.')
    for filename,expected in metadata['files'].items():checked_bytes(directory/filename,expected)
    results={key:checked_json(directory,key)[0]['results'] for key in ('governance','forecast','interval','diagnostic_risk_coverage')}
    print('SAVED OUTER REPORT / FILE CHECKSUMS / REPORTING CODE: PASS')
    print('FIGURES: decisions, group governance, conditional risk, forecast MAE, joint coverage, diagnostic curves')
    if not args.save:
        print('PLAN ONLY. Use --save to export PNG/PDF/SVG; no models or experiments executed.');return
    output=directory/'figures-v1';require(not output.exists(),'Figure version exists; preserve it.')
    figures=render_figures(results['governance'],results['forecast'],results['interval'],results['diagnostic_risk_coverage'])
    output.mkdir(parents=True,exist_ok=False);files={}
    try:
        for stem,figure in figures.items():
            for extension in ('png','pdf','svg'):
                path=output/(stem+'.'+extension);figure.savefig(path,dpi=180)
                files[path.name]=sha(path.read_bytes())
        import matplotlib.pyplot as plt
        for figure in figures.values():plt.close(figure)
        checked_bytes(directory/'metadata.json',digest)
        checksum=save_json(output/'metadata.json',{'purpose':'final_outer_report_figures',
            'created_at_utc':utc_now(),**git_context(),'outer_report_metadata_sha256':digest,
            'source_sha256':sources(),'files':files,'experiments_executed':False,'production_release_authorized':False})
        for filename,expected in files.items():checked_bytes(output/filename,expected)
        print('FIGURES:',output);print('METADATA SHA256:',checksum)
    except Exception:
        # Keep partial figure output for inspection; never overwrite a prior version.
        raise


if __name__=='__main__':main()
