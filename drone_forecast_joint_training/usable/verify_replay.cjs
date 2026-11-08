// 전체 세 임무의 처음/중간/끝, 상태별 점수 보류 및 실제 HTML 스크립트 실행 검사.
const fs=require('node:fs'),vm=require('node:vm');
const path=process.argv[2],text=fs.readFileSync(path,'utf8');
const payload=text.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload),script=text.match(/<script>\s*(const D=[\s\S]*?)<\/script>/)[1];
let draws=0;
const ctx=new Proxy({},{get(o,k){if(k in o)return o[k];return (...args)=>{draws++;for(const x of args)if(typeof x==='number'&&!Number.isFinite(x))throw Error('non-finite draw');}},set(o,k,v){o[k]=v;return true;}});
const ids=['map','case','play','reset','speed','time','clock','maptitle','caseinfo','watchcount','warncount','unkcount','truth','observer','links','diag','data'];
const els=Object.fromEntries(ids.map(id=>[id,{value:'0',checked:false,textContent:'',innerHTML:'',children:[],append(x){this.children.push(x)}}]));
els.data.textContent=payload;els.map.getContext=()=>ctx;els.observer.value='all';els.speed.value='32';
const context=vm.createContext({document:{getElementById:id=>els[id],createElement:()=>({})},requestAnimationFrame:()=>0,console});vm.runInContext(script,context);
let renders=0,decisions=0;
for(let c=0;c<data.cases.length;c++){
 const record=data.cases[c],end=record.positions.length-1;
 if(end!==record.summary.duration_s||record.risk.length!==end+1||record.status.length!==end+1)throw Error('incomplete timeline');
 for(let t=0;t<=end;t++){
  for(const p of record.positions[t])if(p.some(x=>!Number.isFinite(x)||Math.abs(x)>195.1))throw Error('map bounds');
  for(let q=0;q<record.status[t].length;q++){
   const s=record.status[t][q],p=record.risk[t][q];
   if([0,4,5,6].includes(s)&&p!==null)throw Error('unsupported score presented as probability');
   if(p!==null&&(p<0||p>1))throw Error('probability bounds');
  }
 }
 for(const t of [0,1,19,20,Math.floor(end/2),end])for(const truth of [false,true])for(const observer of ['all','0']){
  els.truth.checked=truth;els.observer.value=observer;vm.runInContext(`selected=${c};time=${t};render()`,context);renders++;
  if(!els.clock.textContent.endsWith(end+' 초'))throw Error('timeline label');
 }
 decisions+=end;
}
console.log(JSON.stringify({cases:data.cases.length,full_endpoints:data.cases.map(x=>x.summary.duration_s),render_states:renders,canvas_draw_calls:draws,all_coordinates_finite:true,unsupported_risks_hidden:true}));
