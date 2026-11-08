// 실제 HTML 스크립트와 전체 시계열, 표시 상태의 검사. 브라우저 시각 검수와 구분.
const fs=require('node:fs'),vm=require('node:vm');
const text=fs.readFileSync(process.argv[2],'utf8'),payload=text.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1],data=JSON.parse(payload),script=text.match(/<script>\s*(const D=[\s\S]*?)<\/script>/)[1];let draws=0;
const ctx=new Proxy({},{get(o,k){if(k in o)return o[k];return (...args)=>{draws++;for(const x of args)if(typeof x==='number'&&!Number.isFinite(x))throw Error('non-finite canvas value');}},set(o,k,v){o[k]=v;return true;}});
const ids=['map','case','play','reset','speed','time','clock','maptitle','caseinfo','watchcount','warncount','earlycount','unkcount','truth','observer','links','diag','data','scoreview','showearly','scoretitle'];const els=Object.fromEntries(ids.map(id=>[id,{value:'0',checked:false,textContent:'',innerHTML:'',children:[],append(x){this.children.push(x)}}]));els.data.textContent=payload;els.map.getContext=()=>ctx;els.observer.value='all';els.speed.value='32';els.scoreview.value='risk';const context=vm.createContext({document:{getElementById:id=>els[id],createElement:()=>({})},requestAnimationFrame:()=>0,console});vm.runInContext(script,context);let renders=0;
for(let c=0;c<data.cases.length;c++){
 const r=data.cases[c],end=r.positions.length-1;if(end!==r.summary.duration_s||!r.summary.complete_mission)throw Error('incomplete mission');
 for(const k of ['risk','early','early8','status','prewarning','truth','jammer','connected'])if(r[k].length!==end+1)throw Error('incomplete timeline '+k);
 for(let t=0;t<=end;t++){
  for(const p of r.positions[t])if(p.some(v=>!Number.isFinite(v)||Math.abs(v)>195.1))throw Error('map bounds');
  for(let q=0;q<r.status[t].length;q++){
   for(const k of ['risk','early','early8']){const p=r[k][t][q];if([0,4,5,6,7].includes(r.status[t][q])&&p!==null)throw Error('unsupported risk');if(p!==null&&(!Number.isFinite(p)||p<0||p>1))throw Error('probability bounds');}
   if(r.early[t][q]!==null&&r.early[t][q]>r.risk[t][q]+.000011)throw Error('subset probability');
  }
 }
 for(const t of [0,1,19,20,Math.floor(end/2),end])for(const truth of [false,true])for(const observer of ['all','0'])for(const mode of ['risk','early','early8'])for(const early of [false,true]){els.truth.checked=truth;els.observer.value=observer;els.scoreview.value=mode;els.showearly.checked=early;vm.runInContext(`selected=${c};time=${t};render()`,context);renders++;if(!els.clock.textContent.endsWith(end+' 초'))throw Error('timeline label');}
}
console.log(JSON.stringify({cases:data.cases.length,full_endpoints:data.cases.map(c=>c.summary.duration_s),render_states:renders,canvas_draw_calls:draws,all_coordinates_finite:true,unsupported_probabilities_hidden:true,browser_visual_inspection:false}));
