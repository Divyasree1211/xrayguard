const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs={}, text) => { const node=document.createElementNS(NS,tag); for(const [key,value] of Object.entries(attrs)) node.setAttribute(key,value); if(text!==undefined) node.textContent=text; return node; };
const write = (id,text) => {document.querySelector(id).textContent=text;};
const number = (value,digits=3) => Number.isFinite(value)?value.toFixed(digits):'Unavailable';
const percentage = (value) => Number.isFinite(value)?`${(100*value).toFixed(1)}%`:'Unavailable';
const label = (value) => value.replaceAll('_',' ');

async function loadResults() {
  try {
    const response=await fetch('/api/results'); const data=await response.json();
    if(!response.ok) throw new Error(data.error || 'Evaluation unavailable.');
    const m=data.classification;
    write('#results-status',`Model run ${data.model_run} · Original saved test evaluation`);
    write('#mean-auroc',number(m.mean_auroc));
    write('#auroc-ci',`95% patient-bootstrap CI ${number(m.mean_auroc_ci[0])}–${number(m.mean_auroc_ci[1])}`);
    const s=data.selection?.splits;
    write('#split-description',s?`${s.train.patients.toLocaleString()} training patients, ${s.val.patients.toLocaleString()} validation patients, and ${s.test.patients.toLocaleString()} test patients. One image per patient.`:'Split details unavailable.');
    write('#test-description',`${m.n_images.toLocaleString()} test images from ${m.n_patients.toLocaleString()} patients. Training and validation are label-enriched. Test patients come from a sample of the official NIH test pool.`);
    const svg=document.querySelector('#auroc-chart');
    const rows=Object.entries(m.per_class).sort((a,b)=>b[1].auroc-a[1].auroc);
    for(let tick=0;tick<=4;tick++) {
      const x=230+tick/4*650;
      svg.append(el('line',{x1:x,x2:x,y1:38,y2:545,class:'chart-gridline'}));
      svg.append(el('text',{x,y:24,'text-anchor':'middle',class:'chart-tick'},(tick/4).toFixed(2)));
    }
    rows.forEach(([name,metric],i)=>{
      const y=67+i*35;
      svg.append(el('text',{x:212,y:y+5,'text-anchor':'end',class:'chart-label'},label(name)));
      svg.append(el('rect',{x:230,y:y-8,width:metric.auroc*650,height:16,rx:3,class:'chart-flag'}));
      if(metric.auroc_ci?.every(Number.isFinite)) {
        const [lo,hi]=metric.auroc_ci.map(value=>230+value*650);
        svg.append(el('line',{x1:lo,x2:hi,y1:y,y2:y,class:'chart-threshold'}));
        for(const x of [lo,hi]) svg.append(el('line',{x1:x,x2:x,y1:y-6,y2:y+6,class:'chart-threshold'}));
      }
      svg.append(el('text',{x:900,y:y+5,class:'chart-value'},`${number(metric.auroc)} (n+ ${metric.n_pos})`));
      const tr=document.createElement('tr');
      for(const value of [label(name),metric.n_pos,number(metric.auroc),number(metric.auprc),percentage(metric.precision),percentage(metric.recall)]) {const cell=document.createElement('td');cell.textContent=value;tr.append(cell);}
      document.querySelector('#metrics-rows').append(tr);
    });
    svg.append(el('text',{x:230,y:575,class:'chart-tick'},'AUROC with 95% confidence interval. n+ is the number of positive cases.'));
    const p=m.per_class.Pneumonia;
    const tp=Math.round(p.recall*p.n_pos),fn=p.n_pos-tp,tn=Math.round(p.specificity*(m.n_images-p.n_pos)),fp=m.n_images-p.n_pos-tn;
    write('#pneumonia-description',`Precision ${percentage(p.precision)} and recall ${percentage(p.recall)} on ${p.n_pos} NIH-positive cases.`);
    for(const [name,count] of [['Correct positive flags',tp],['Missed positives',fn],['False positive flags',fp],['Correct negatives',tn]]) {
      const div=document.createElement('div'); const value=document.createElement('strong');value.textContent=count;const description=document.createElement('span');description.textContent=name;div.append(value,description);document.querySelector('#confusion-counts').append(div);
    }
    for(const [name,raw,calibrated] of [['ECE',m.calibration.ece_raw,m.calibration.ece_T],['Brier score',m.calibration.brier_raw,m.calibration.brier_T]]) {
      const tr=document.createElement('tr');for(const value of [name,number(raw,4),number(calibrated,4)]) {const td=document.createElement('td');td.textContent=value;tr.append(td);} document.querySelector('#calibration-rows').append(tr);
    }
    const history=data.training_history;
    if(history.length) {
      const training=document.querySelector('#training-chart');
      for(let tick=0;tick<=4;tick++) {const y=270-tick*50;training.append(el('line',{x1:65,x2:550,y1:y,y2:y,class:'chart-gridline'}));training.append(el('text',{x:52,y:y+5,'text-anchor':'end',class:'chart-tick'},(.6+tick*.1).toFixed(1)));}
      const coords=history.map((row,i)=>[65+i/Math.max(1,history.length-1)*485,270-(row.val_mean_auroc-.6)/.4*200]);
      training.append(el('polyline',{points:coords.map(point=>point.join(',')).join(' '),fill:'none',stroke:'#16816e','stroke-width':3}));
      coords.forEach(([x,y],i)=>{training.append(el('circle',{cx:x,cy:y,r:5,fill:'#16816e'}));training.append(el('text',{x,y:300,'text-anchor':'middle',class:'chart-tick'},history[i].epoch));});
      training.append(el('text',{x:260,y:330,class:'chart-tick'},'Epoch'));
      const best=history.reduce((a,b)=>a.val_mean_auroc>b.val_mean_auroc?a:b);
      write('#training-description',`Best validation AUROC ${number(best.val_mean_auroc,4)} at epoch ${best.epoch}. Training history totals ${history.reduce((sum,row)=>sum+row.minutes,0).toFixed(1)} minutes. Test data did not select this checkpoint.`);
    } else write('#training-description','Training history unavailable.');
    const stress=data.current_rule_stress_test;
    write('#stress-description',stress?`${stress.n_reports} evidence records and ${stress.overall.n.toLocaleString()} injected errors. Detection ${percentage(stress.overall.detection_recall)}, repair ${percentage(stress.overall.repair_rate)}, clean-report false alarms ${percentage(stress.clean_false_alarm_rate)}.`:'Current verifier stress-test results unavailable.');
    write('#report-version',data.report_metrics_version);
    const reports=data.original_report_metrics?.['tiered_mc (ours)'];
    write('#original-report-description',reports?`Earlier template results: micro-F1 ${number(reports.micro_f1)}, positive-mention precision ${percentage(reports.precision)}, and omission ${percentage(reports.omission_rate)} against NIH labels. Wording consistency does not guarantee correct image interpretation.`:'Original report metrics unavailable.');
    document.querySelector('#export-metrics').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='xrayguard-evaluation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    document.querySelector('#evaluation-content').hidden=false;
  } catch(error) {write('#results-status',error.message);}
}
loadResults();
