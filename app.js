/* SLP Interactive — demos backed by the tested ES modules in ./src/. */
import { knnPredict } from './src/knn.js';
import { assign as kmAssign, updateCentroids as kmUpdate, wss as kmWss, kmeansPlusPlusInit } from './src/kmeans.js';
import { trainStep as logregStep } from './src/logreg.js';
import { bootstrap as bagBootstrap } from './src/ensemble.js';

(() => {
'use strict';

/* ============ Router ============ */
const nav = document.getElementById('nav');
const sections = [...document.querySelectorAll('.section')];
function go(id){
  sections.forEach(s => s.classList.toggle('active', s.id===id));
  document.querySelectorAll('[data-go]').forEach(a => a.classList.toggle('active', a.dataset.go===id));
  // re-fit canvases on activation
  if(MOUNTED[id]) MOUNTED[id]();
  window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('[data-go]').forEach(a => a.addEventListener('click', () => go(a.dataset.go)));

/* ============ Utilities ============ */
const rng = (() => { let s=1234567; return () => { s = (s*48271)%2147483647; return s/2147483647; }; })();
const randn = () => { let u=0,v=0; while(!u) u=rng(); while(!v) v=rng(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
function fitCanvas(c){
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(1, Math.floor(r.width*dpr));
  c.height = Math.max(1, Math.floor((c.clientHeight||r.height||400)*dpr));
  const g = c.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  return {g, w:c.clientWidth, h:c.clientHeight||r.height||400};
}
function gridBg(g,w,h,step=40){
  g.fillStyle='#0f1623'; g.fillRect(0,0,w,h);
  g.strokeStyle='#1c2638'; g.lineWidth=1; g.beginPath();
  for(let x=0;x<=w;x+=step){g.moveTo(x,0);g.lineTo(x,h);}
  for(let y=0;y<=h;y+=step){g.moveTo(0,y);g.lineTo(w,y);}
  g.stroke();
}
function classColor(c){ return c===0||c===-1 ? '#7cc4ff' : c===1 ? '#ff7c9c' : ['#7cc4ff','#ff7c9c','#7cffb2','#ffb86b','#9d7cff'][c%5]; }
function classColorAlpha(c,a){
  const map = {0:`rgba(124,196,255,${a})`,1:`rgba(255,124,156,${a})`, 2:`rgba(124,255,178,${a})`, 3:`rgba(255,184,107,${a})`, 4:`rgba(157,124,255,${a})`};
  if(c===-1) return map[0];
  return map[c] || map[0];
}
function drawPoint(g,x,y,c,r=5,outline=true){
  g.fillStyle = classColor(c);
  g.beginPath(); g.arc(x,y,r,0,Math.PI*2); g.fill();
  if(outline){ g.strokeStyle='rgba(0,0,0,.5)'; g.lineWidth=1; g.stroke(); }
}
function bindClick(c, fn){
  c.addEventListener('click', e => {
    const r = c.getBoundingClientRect();
    fn(e.clientX-r.left, e.clientY-r.top, e);
  });
}
function bindDrag(c, hitFn, moveFn, upFn){
  let dragIdx=-1;
  c.addEventListener('mousedown', e=>{
    const r=c.getBoundingClientRect();
    dragIdx = hitFn(e.clientX-r.left, e.clientY-r.top);
  });
  window.addEventListener('mousemove', e=>{
    if(dragIdx<0) return;
    const r=c.getBoundingClientRect();
    moveFn(dragIdx, e.clientX-r.left, e.clientY-r.top);
  });
  window.addEventListener('mouseup', ()=>{ if(dragIdx>=0){ upFn&&upFn(dragIdx); dragIdx=-1; }});
}
// Matrix utils for linreg etc.
function transpose(A){ const r=A.length,c=A[0].length; const T=Array.from({length:c},()=>new Array(r)); for(let i=0;i<r;i++)for(let j=0;j<c;j++)T[j][i]=A[i][j]; return T; }
function matmul(A,B){ const r=A.length,c=B[0].length,k=B.length; const C=Array.from({length:r},()=>new Array(c).fill(0));
  for(let i=0;i<r;i++) for(let kk=0;kk<k;kk++){ const a=A[i][kk]; for(let j=0;j<c;j++) C[i][j]+=a*B[kk][j]; } return C; }
function matvec(A,x){ const r=A.length; const y=new Array(r).fill(0); for(let i=0;i<r;i++){ let s=0; const Ai=A[i]; for(let j=0;j<x.length;j++) s+=Ai[j]*x[j]; y[i]=s; } return y; }
function inv(A){ // Gauss-Jordan
  const n=A.length;
  const M=A.map((row,i)=>{const r=row.slice(); const id=new Array(n).fill(0); id[i]=1; return r.concat(id);});
  for(let i=0;i<n;i++){
    let p=i; for(let r=i+1;r<n;r++) if(Math.abs(M[r][i])>Math.abs(M[p][i])) p=r;
    if(Math.abs(M[p][i])<1e-12) M[p][i]+=1e-9;
    [M[i],M[p]]=[M[p],M[i]];
    const d=M[i][i]; for(let j=0;j<2*n;j++) M[i][j]/=d;
    for(let r=0;r<n;r++) if(r!==i){ const f=M[r][i]; for(let j=0;j<2*n;j++) M[r][j]-=f*M[i][j]; }
  }
  return M.map(row=>row.slice(n));
}
function solve(A,b){ return matvec(inv(A), b); }

const MOUNTED = {};

/* ====================================================
   k-NN
==================================================== */
(function(){
  const c=document.getElementById('knn-c');
  let pts=[]; // {x,y,c}
  const k=document.getElementById('knn-k'),kv=document.getElementById('knn-kv');
  const distSel=document.getElementById('knn-dist'),wSel=document.getElementById('knn-w'),clsSel=document.getElementById('knn-cls');
  const W=()=>c.clientWidth, H=()=>c.clientHeight||420;
  function predict(x,y){
    // Delegates to the tested k-NN core in ./src/knn.js.
    return knnPredict(pts, {x,y}, {
      k: +k.value,
      metric: distSel.value==='l1' ? 'l1' : 'l2',
      weight: wSel.value==='inv' ? 'inverse' : 'uniform',
    });
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(pts.length){
      // Decision region (coarse pixels)
      const step=8;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        const p=predict(x,y); if(p<0) continue;
        g.fillStyle = classColorAlpha(p,.18); g.fillRect(x,y,step,step);
      }
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,6));
    // legend
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`n=${pts.length}  k=${k.value}  ${distSel.value.toUpperCase()}  ${wSel.value}`,10,16);
  }
  function demo(){ pts=[]; const W0=W(),H0=H();
    for(let i=0;i<25;i++) pts.push({x:W0*.25+randn()*40, y:H0*.6+randn()*40, c:0});
    for(let i=0;i<25;i++) pts.push({x:W0*.65+randn()*45, y:H0*.4+randn()*45, c:1});
    draw();
  }
  bindClick(c,(x,y)=>{ pts.push({x,y,c:+clsSel.value}); draw(); });
  [k,distSel,wSel].forEach(el=>el.addEventListener('input',()=>{kv.textContent=k.value; draw();}));
  document.getElementById('knn-demo').onclick=demo;
  document.getElementById('knn-clear').onclick=()=>{pts=[];draw();};
  MOUNTED.knn = () => { if(!pts.length) demo(); else draw(); };
})();

/* ====================================================
   Linear Regression (polynomial OLS)
==================================================== */
(function(){
  const c=document.getElementById('lr-c');
  const d=document.getElementById('lr-d'),dv=document.getElementById('lr-dv');
  const mseEl=document.getElementById('lr-mse'),r2El=document.getElementById('lr-r2');
  let pts=[]; let drag=-1;
  function fit(){
    const n=pts.length, deg=+d.value;
    if(n<2) return null;
    const W=c.clientWidth, H=c.clientHeight||400;
    const X=pts.map(p=>{ const x=(p.x/W)*2-1; const row=[]; for(let j=0;j<=deg;j++) row.push(Math.pow(x,j)); return row; });
    const y=pts.map(p=> 1-(p.y/H)*2 );
    const XT=transpose(X), XTX=matmul(XT,X);
    for(let i=0;i<XTX.length;i++) XTX[i][i]+=1e-8;
    const w=solve(XTX, matvec(XT,y));
    // metrics
    const ypred=matvec(X,w);
    let mse=0, ymean=y.reduce((a,b)=>a+b,0)/n, sst=0, ssr=0;
    for(let i=0;i<n;i++){ const e=y[i]-ypred[i]; mse+=e*e; sst+=(y[i]-ymean)**2; ssr+=e*e; }
    mse/=n;
    const r2 = sst>0 ? 1-ssr/sst : 0;
    return {w,mse,r2,W,H};
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    // axes
    g.strokeStyle='#2a3a63'; g.lineWidth=1; g.beginPath(); g.moveTo(0,h/2);g.lineTo(w,h/2); g.moveTo(w/2,0);g.lineTo(w/2,h); g.stroke();
    const fitObj = fit();
    if(fitObj){
      const {w:wv,W,H} = fitObj;
      g.strokeStyle='#7cffb2'; g.lineWidth=2; g.beginPath();
      for(let px=0;px<w;px+=2){
        const x=(px/w)*2-1; let yhat=0; for(let j=0;j<wv.length;j++) yhat+=wv[j]*Math.pow(x,j);
        const py=(1-yhat)/2 * h;
        if(px===0) g.moveTo(px,py); else g.lineTo(px,py);
      }
      g.stroke();
      mseEl.textContent = fitObj.mse.toFixed(3);
      r2El.textContent = fitObj.r2.toFixed(3);
    } else { mseEl.textContent='—'; r2El.textContent='—'; }
    pts.forEach((p,i)=>drawPoint(g,p.x,p.y,0,5));
  }
  c.addEventListener('mousedown',e=>{
    const r=c.getBoundingClientRect(); const x=e.clientX-r.left,y=e.clientY-r.top;
    let hit=-1; pts.forEach((p,i)=>{ if(Math.hypot(p.x-x,p.y-y)<10) hit=i; });
    if(hit<0){ pts.push({x,y}); draw(); } else drag=hit;
  });
  window.addEventListener('mousemove',e=>{
    if(drag<0) return; const r=c.getBoundingClientRect();
    pts[drag].x=e.clientX-r.left; pts[drag].y=e.clientY-r.top; draw();
  });
  window.addEventListener('mouseup',()=>drag=-1);
  d.addEventListener('input',()=>{dv.textContent=d.value; draw();});
  document.getElementById('lr-demo').onclick=()=>{
    pts=[]; const W=c.clientWidth,H=c.clientHeight||400;
    for(let i=0;i<25;i++){
      const x=(i/24)*2-1;
      const t=Math.sin(x*Math.PI*1.5)*0.6+randn()*0.12;
      pts.push({x:((x+1)/2)*W, y:((1-t)/2)*H});
    }
    draw();
  };
  document.getElementById('lr-clear').onclick=()=>{pts=[];draw();};
  MOUNTED.linreg=()=>{ if(!pts.length) document.getElementById('lr-demo').onclick(); else draw(); };
})();

/* ====================================================
   Loss Functions plot
==================================================== */
(function(){
  const c=document.getElementById('loss-c');
  const d=document.getElementById('loss-d'),dv=document.getElementById('loss-dv');
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    const x0=w/2, y0=h*.85, sx=w/8, sy=h/4;
    g.strokeStyle='#2a3a63'; g.beginPath(); g.moveTo(0,y0);g.lineTo(w,y0); g.moveTo(x0,0);g.lineTo(x0,h); g.stroke();
    g.fillStyle='#8b96a8'; g.font='11px monospace';
    g.fillText('error', w-40, y0-4); g.fillText('loss', x0+4, 12);
    const delta=+d.value;
    function plot(fn,col,style='solid'){
      g.strokeStyle=col; g.lineWidth=2;
      if(style==='dash') g.setLineDash([4,3]); else g.setLineDash([]);
      g.beginPath();
      for(let px=0;px<w;px++){
        const x=(px-x0)/sx; const y=fn(x); const py=y0-y*sy;
        if(px===0) g.moveTo(px,py); else g.lineTo(px,py);
      }
      g.stroke(); g.setLineDash([]);
    }
    plot(x=>x*x, '#7cc4ff'); // L2
    plot(x=>Math.abs(x), '#ff7c9c'); // L1
    plot(x=>{ const a=Math.abs(x); return a<=delta? .5*x*x : delta*(a-.5*delta); }, '#ffb86b'); // Huber
    // classification (margin x = y*f)
    plot(x=>x<0?1:0, '#7cffb2', 'dash'); // 0/1
    plot(x=>Math.max(0,1-x), '#9d7cff'); // hinge
    plot(x=>Math.log(1+Math.exp(-x))/Math.log(2), '#e6edf3'); // log
  }
  d.addEventListener('input',()=>{dv.textContent=(+d.value).toFixed(2); draw();});
  MOUNTED.loss=draw;
})();

/* ====================================================
   Regularization
==================================================== */
(function(){
  const c=document.getElementById('reg-c'), cb=document.getElementById('reg-bars');
  const lr=document.getElementById('reg-l'),lv=document.getElementById('reg-lv');
  const tSel=document.getElementById('reg-type'),dEl=document.getElementById('reg-d'),dv=document.getElementById('reg-dv');
  let pts=[];
  function makeData(){
    pts=[]; const W=c.clientWidth, H=c.clientHeight||400;
    for(let i=0;i<30;i++){
      const x=(i/29)*2-1;
      const t=Math.sin(x*Math.PI)*0.5+0.2*x+randn()*0.15;
      pts.push({x:((x+1)/2)*W, y:((1-t)/2)*H});
    }
    fitDraw();
  }
  function design(){
    const deg=+dEl.value, W=c.clientWidth;
    return pts.map(p=>{ const x=(p.x/W)*2-1; const row=[]; for(let j=0;j<=deg;j++) row.push(Math.pow(x,j)); return row; });
  }
  function ridgeFit(){
    const X=design(), n=X.length; if(!n) return null;
    const H0=c.clientHeight||400;
    const y=pts.map(p=>1-(p.y/H0)*2);
    const lam=Math.pow(10, +lr.value);
    const XT=transpose(X), XTX=matmul(XT,X);
    for(let i=0;i<XTX.length;i++) XTX[i][i]+=(i===0?0:lam); // don't penalise intercept
    return solve(XTX, matvec(XT,y));
  }
  function lassoFit(){
    // Coordinate descent
    const X=design(), n=X.length; if(!n) return null;
    const H0=c.clientHeight||400;
    const y=pts.map(p=>1-(p.y/H0)*2);
    const p=X[0].length, w=new Array(p).fill(0);
    const lam=Math.pow(10, +lr.value);
    // Precompute column squared norms
    const colsq=new Array(p).fill(0);
    for(let j=0;j<p;j++) for(let i=0;i<n;i++) colsq[j]+=X[i][j]*X[i][j];
    const resid=y.slice();
    for(let it=0;it<150;it++){
      let maxd=0;
      for(let j=0;j<p;j++){
        let rho=0; for(let i=0;i<n;i++) rho+=X[i][j]*(resid[i]+X[i][j]*w[j]);
        let wj;
        if(j===0){ wj = rho/colsq[j]; } // intercept unpenalised
        else {
          if(rho < -lam) wj=(rho+lam)/colsq[j];
          else if(rho > lam) wj=(rho-lam)/colsq[j];
          else wj=0;
        }
        const dw=wj-w[j]; if(dw){
          for(let i=0;i<n;i++) resid[i]-=X[i][j]*dw;
          w[j]=wj; if(Math.abs(dw)>maxd) maxd=Math.abs(dw);
        }
      }
      if(maxd<1e-5) break;
    }
    return w;
  }
  function fitDraw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    g.strokeStyle='#2a3a63'; g.beginPath(); g.moveTo(0,h/2);g.lineTo(w,h/2); g.moveTo(w/2,0);g.lineTo(w/2,h); g.stroke();
    if(!pts.length) return;
    const coef = tSel.value==='lasso' ? lassoFit() : ridgeFit();
    // Plot OLS faint for comparison
    g.strokeStyle='#ff7c9c'; g.lineWidth=1; g.setLineDash([4,3]); g.beginPath();
    const ols = (function(){
      const X=design(); const XT=transpose(X), XTX=matmul(XT,X);
      const H0=c.clientHeight||400;
      const y=pts.map(p=>1-(p.y/H0)*2);
      for(let i=0;i<XTX.length;i++) XTX[i][i]+=1e-8;
      return solve(XTX, matvec(XT,y));
    })();
    for(let px=0;px<w;px+=2){
      const x=(px/w)*2-1; let yh=0; for(let j=0;j<ols.length;j++) yh+=ols[j]*Math.pow(x,j);
      const py=(1-yh)/2*h;
      if(px===0) g.moveTo(px,py); else g.lineTo(px,py);
    }
    g.stroke(); g.setLineDash([]);
    // Regularised fit
    g.strokeStyle='#7cffb2'; g.lineWidth=2; g.beginPath();
    for(let px=0;px<w;px+=2){
      const x=(px/w)*2-1; let yh=0; for(let j=0;j<coef.length;j++) yh+=coef[j]*Math.pow(x,j);
      const py=(1-yh)/2*h;
      if(px===0) g.moveTo(px,py); else g.lineTo(px,py);
    }
    g.stroke();
    pts.forEach(p=>drawPoint(g,p.x,p.y,0,4));
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`${tSel.value}  λ=10^${(+lr.value).toFixed(1)}  deg=${dEl.value}`, 10, 16);
    // bars
    const {g:gb, w:wb, h:hb} = fitCanvas(cb);
    gb.fillStyle='#0f1623'; gb.fillRect(0,0,wb,hb);
    const bw = wb/coef.length;
    const maxA = Math.max(.1, ...coef.map(c=>Math.abs(c)));
    coef.forEach((cc,i)=>{
      const x=i*bw, hb2=(Math.abs(cc)/maxA)*(hb*.45);
      gb.fillStyle = cc>=0?'#7cc4ff':'#ff7c9c';
      const y0=hb/2;
      if(cc>=0) gb.fillRect(x+2, y0-hb2, bw-4, hb2);
      else gb.fillRect(x+2, y0, bw-4, hb2);
      gb.fillStyle='#8b96a8'; gb.font='10px monospace'; gb.fillText('w'+i, x+4, hb-4);
    });
    gb.strokeStyle='#2a3a63'; gb.beginPath(); gb.moveTo(0,hb/2);gb.lineTo(wb,hb/2); gb.stroke();
  }
  [lr,tSel,dEl].forEach(el=>el.addEventListener('input',()=>{
    lv.textContent=(+lr.value).toFixed(1); dv.textContent=dEl.value; fitDraw();
  }));
  document.getElementById('reg-demo').onclick=makeData;
  MOUNTED.reg = () => { if(!pts.length) makeData(); else fitDraw(); };
})();

/* ====================================================
   Bias-Variance
==================================================== */
(function(){
  const c=document.getElementById('bv-c'), cc=document.getElementById('bv-curve');
  const dEl=document.getElementById('bv-d'),dv=document.getElementById('bv-dv');
  const nEl=document.getElementById('bv-n'),nv=document.getElementById('bv-nv');
  const sEl=document.getElementById('bv-s'),sv=document.getElementById('bv-sv');
  function trueF(x){ return Math.sin(x*Math.PI); }
  function sample(N,noise){
    const pts=[]; for(let i=0;i<N;i++){ const x=(rng()*2-1); pts.push({x,y:trueF(x)+randn()*noise}); } return pts;
  }
  function polyFit(pts,deg){
    const X=pts.map(p=>{ const r=[]; for(let j=0;j<=deg;j++) r.push(Math.pow(p.x,j)); return r;});
    const y=pts.map(p=>p.y); const XT=transpose(X), XTX=matmul(XT,X);
    for(let i=0;i<XTX.length;i++) XTX[i][i]+=1e-8;
    return solve(XTX, matvec(XT,y));
  }
  function evalPoly(w,x){ let y=0; for(let j=0;j<w.length;j++) y+=w[j]*Math.pow(x,j); return y; }
  function drawFits(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    const deg=+dEl.value, noise=+nEl.value, N=+sEl.value;
    function px(x){ return (x+1)/2*w; }
    function py(y){ return (1-(y+1.2)/2.4)*h; }
    // true f
    g.strokeStyle='#ffb86b'; g.lineWidth=2; g.beginPath();
    for(let i=0;i<=200;i++){ const x=-1+i/100; const Y=trueF(x); if(i===0) g.moveTo(px(x),py(Y)); else g.lineTo(px(x),py(Y)); }
    g.stroke();
    // many fits
    g.strokeStyle='rgba(124,196,255,.4)'; g.lineWidth=1;
    for(let m=0;m<25;m++){
      const data=sample(N,noise); const wcoef=polyFit(data,deg);
      g.beginPath();
      for(let i=0;i<=200;i++){ const x=-1+i/100; const Y=evalPoly(wcoef,x); if(i===0) g.moveTo(px(x),py(Y)); else g.lineTo(px(x),py(Y)); }
      g.stroke();
    }
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`degree=${deg}  σ=${noise.toFixed(2)}  N=${N}`,10,16);
  }
  function drawCurve(){
    const {g,w,h}=fitCanvas(cc); gridBg(g,w,h);
    const noise=+nEl.value, N=+sEl.value;
    const maxDeg=15;
    const train=[], test=[];
    // Pre-compute average over multiple trials
    for(let deg=1;deg<=maxDeg;deg++){
      let tr=0, te=0, trials=10;
      for(let t=0;t<trials;t++){
        const data=sample(N,noise);
        const wcoef=polyFit(data,deg);
        // train MSE
        let s=0; for(const p of data){ const e=p.y-evalPoly(wcoef,p.x); s+=e*e; } tr+=s/data.length;
        // test (fresh)
        const testD=sample(200,noise); s=0; for(const p of testD){ const e=p.y-evalPoly(wcoef,p.x); s+=e*e; } te+=s/testD.length;
      }
      train.push(tr/trials); test.push(te/trials);
    }
    const maxV=Math.max(...test, ...train, 0.3);
    function px(i){ return 20+(i-1)/(maxDeg-1)*(w-40); }
    function py(v){ return h-30-Math.min(v,maxV)/maxV*(h-60); }
    g.strokeStyle='#2a3a63'; g.beginPath(); g.moveTo(20,h-30); g.lineTo(w-20,h-30); g.moveTo(20,h-30); g.lineTo(20,20); g.stroke();
    g.fillStyle='#8b96a8'; g.font='11px monospace'; g.fillText('complexity →', w-90, h-10); g.fillText('error', 22, 18);
    function line(arr,col){
      g.strokeStyle=col; g.lineWidth=2; g.beginPath();
      arr.forEach((v,i)=>{ const x=px(i+1),y=py(v); if(i===0) g.moveTo(x,y); else g.lineTo(x,y); }); g.stroke();
      arr.forEach((v,i)=>{ g.fillStyle=col; g.beginPath(); g.arc(px(i+1),py(v),3,0,Math.PI*2); g.fill(); });
    }
    line(train,'#7cffb2'); line(test,'#ff7c9c');
    // mark current degree
    const deg=+dEl.value;
    g.strokeStyle='#9d7cff'; g.setLineDash([3,3]); g.beginPath(); g.moveTo(px(deg),20); g.lineTo(px(deg),h-30); g.stroke(); g.setLineDash([]);
    g.fillStyle='#7cffb2'; g.fillText('train', w-60, 30); g.fillStyle='#ff7c9c'; g.fillText('test', w-60, 46);
  }
  [dEl,nEl,sEl].forEach(el=>el.addEventListener('input',()=>{
    dv.textContent=dEl.value; nv.textContent=(+nEl.value).toFixed(2); sv.textContent=sEl.value; drawFits(); drawCurve();
  }));
  document.getElementById('bv-resamp').onclick=()=>{ drawFits(); drawCurve(); };
  MOUNTED.bv = () => { drawFits(); drawCurve(); };
})();

/* ====================================================
   Gradient Descent
==================================================== */
(function(){
  const c=document.getElementById('gd-c');
  const lrEl=document.getElementById('gd-lr'),lrv=document.getElementById('gd-lrv');
  const mEl=document.getElementById('gd-m'),mv=document.getElementById('gd-m');
  const surfSel=document.getElementById('gd-surf');
  let surf='bowl', path=[{x:-1.8,y:1.4}], vel={x:0,y:0};
  const surfs={
    bowl:(x,y)=>(0.4*x*x+1.6*y*y),
    rosen:(x,y)=>{ const a=1,b=20; return (a-x)**2 + b*(y-x*x)**2; },
    saddle:(x,y)=>(x*x - y*y)*0.4,
    ravine:(x,y)=>(0.05*x*x + 4*y*y)
  };
  function grad(x,y){
    const eps=1e-3, f=surfs[surf];
    return {gx:(f(x+eps,y)-f(x-eps,y))/(2*eps), gy:(f(x,y+eps)-f(x,y-eps))/(2*eps)};
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    // Sample to find range
    const f=surfs[surf];
    let vmin=Infinity,vmax=-Infinity;
    const N=80; const vals=Array.from({length:N},()=>new Array(N));
    for(let i=0;i<N;i++) for(let j=0;j<N;j++){
      const x=(i/(N-1))*5-2.5, y=(j/(N-1))*4-2;
      const v=f(x,y); vals[i][j]=v; if(v<vmin)vmin=v; if(v>vmax)vmax=v;
    }
    // heatmap
    const cellW=w/N, cellH=h/N;
    for(let i=0;i<N;i++) for(let j=0;j<N;j++){
      const t=(vals[i][j]-vmin)/(vmax-vmin+1e-9);
      const r=Math.floor(20+t*200), b=Math.floor(120-t*60), gC=Math.floor(50+t*40);
      g.fillStyle=`rgb(${r},${gC},${b})`; g.fillRect(i*cellW, (N-1-j)*cellH, cellW+1, cellH+1);
    }
    // contour lines (a few levels)
    g.strokeStyle='rgba(255,255,255,.18)'; g.lineWidth=1;
    for(let lvl=1;lvl<=6;lvl++){
      const target=vmin+(vmax-vmin)*lvl/7;
      for(let i=0;i<N-1;i++) for(let j=0;j<N-1;j++){
        const v=vals[i][j];
        if(((v-target)*(vals[i+1][j]-target))<0 || ((v-target)*(vals[i][j+1]-target))<0){
          g.fillStyle='rgba(255,255,255,.12)'; g.fillRect(i*cellW,(N-1-j)*cellH,cellW,cellH);
        }
      }
    }
    function toScr(x,y){ return {sx:(x+2.5)/5*w, sy:(1-(y+2)/4)*h}; }
    // path
    g.strokeStyle='#ffb86b'; g.lineWidth=2; g.beginPath();
    path.forEach((p,i)=>{ const s=toScr(p.x,p.y); if(i===0) g.moveTo(s.sx,s.sy); else g.lineTo(s.sx,s.sy); });
    g.stroke();
    path.forEach((p,i)=>{ const s=toScr(p.x,p.y); g.fillStyle=i===path.length-1?'#7cffb2':'#ffb86b'; g.beginPath(); g.arc(s.sx,s.sy,i===path.length-1?6:3,0,Math.PI*2); g.fill(); });
    g.fillStyle='#fff'; g.font='12px monospace';
    g.fillText(`step ${path.length-1}  f=${f(path[path.length-1].x,path[path.length-1].y).toFixed(4)}`,10,16);
  }
  function step(){
    const last=path[path.length-1], lr=+lrEl.value, beta=+mEl.value;
    const {gx,gy}=grad(last.x,last.y);
    vel.x = beta*vel.x - lr*gx;
    vel.y = beta*vel.y - lr*gy;
    const nx=clamp(last.x+vel.x,-2.4,2.4), ny=clamp(last.y+vel.y,-1.9,1.9);
    path.push({x:nx,y:ny}); draw();
  }
  let running=false;
  function reset(){ path=[{x:-1.8,y:1.4}]; vel={x:0,y:0}; draw(); }
  bindClick(c,(sx,sy)=>{ const w=c.clientWidth,h=c.clientHeight||430; const x=sx/w*5-2.5, y=(1-sy/h)*4-2; path=[{x,y}]; vel={x:0,y:0}; draw(); });
  lrEl.addEventListener('input',()=>{lrv.textContent=(+lrEl.value).toFixed(3);});
  mEl.addEventListener('input',()=>{mv.textContent=(+mEl.value).toFixed(2);});
  surfSel.addEventListener('change',()=>{surf=surfSel.value; reset();});
  document.getElementById('gd-step').onclick=step;
  document.getElementById('gd-reset').onclick=reset;
  document.getElementById('gd-run').onclick=async()=>{ for(let i=0;i<60;i++){ step(); await new Promise(r=>setTimeout(r,30)); } };
  MOUNTED.gd = () => draw();
})();

/* ====================================================
   Softmax + SGD
==================================================== */
(function(){
  const c=document.getElementById('sm-c');
  const kEl=document.getElementById('sm-k'),kv=document.getElementById('sm-kv');
  const lrEl=document.getElementById('sm-lr'),lrv=document.getElementById('sm-lrv');
  const bEl=document.getElementById('sm-b'),bv=document.getElementById('sm-bv');
  const lossEl=document.getElementById('sm-loss'),accEl=document.getElementById('sm-acc');
  let pts=[], W=null, K=3, currentCls=0, timer=null;
  function initW(){
    K=+kEl.value;
    W = Array.from({length:K},()=>[randn()*.1,randn()*.1,randn()*.1]); // x,y,bias
  }
  function softmax(x,y){
    const logits=W.map(w=>w[0]*x+w[1]*y+w[2]);
    const m=Math.max(...logits); const ex=logits.map(l=>Math.exp(l-m));
    const s=ex.reduce((a,b)=>a+b,0); return ex.map(e=>e/s);
  }
  function makeData(){
    pts=[]; const ww=c.clientWidth||400, hh=c.clientHeight||420;
    for(let k=0;k<K;k++){
      const cx=ww*(.2+.6*(k+1)/(K+1)), cy=hh*(.5+.2*Math.sin(k));
      for(let i=0;i<30;i++) pts.push({x:cx+randn()*30, y:cy+randn()*30, c:k});
    }
  }
  function trainStep(){
    if(!pts.length) return;
    const B=+bEl.value, lr=+lrEl.value, ww=c.clientWidth,hh=c.clientHeight||420;
    let loss=0;
    for(let it=0;it<5;it++){
      // mini-batch
      const grads = W.map(()=>[0,0,0]);
      for(let b=0;b<B;b++){
        const p=pts[Math.floor(rng()*pts.length)];
        const x=(p.x/ww)*2-1, y=(p.y/hh)*2-1;
        const probs=softmax(x,y);
        loss -= Math.log(probs[p.c]+1e-9);
        for(let k=0;k<K;k++){
          const err=probs[k]-(k===p.c?1:0);
          grads[k][0]+=err*x; grads[k][1]+=err*y; grads[k][2]+=err;
        }
      }
      for(let k=0;k<K;k++) for(let j=0;j<3;j++) W[k][j]-=lr*grads[k][j]/B;
    }
    // accuracy
    let correct=0;
    for(const p of pts){
      const x=(p.x/ww)*2-1, y=(p.y/hh)*2-1;
      const pr=softmax(x,y); let am=0; for(let k=1;k<K;k++) if(pr[k]>pr[am]) am=k;
      if(am===p.c) correct++;
    }
    lossEl.textContent=(loss/5).toFixed(3);
    accEl.textContent=(correct/pts.length*100).toFixed(1)+'%';
    draw();
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(W){
      const step=8;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        const xn=(x/w)*2-1, yn=(y/h)*2-1;
        const pr=softmax(xn,yn);
        let am=0; for(let k=1;k<K;k++) if(pr[k]>pr[am]) am=k;
        g.fillStyle=classColorAlpha(am, .15+pr[am]*.2);
        g.fillRect(x,y,step,step);
      }
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
  }
  bindClick(c,(x,y)=>{ pts.push({x,y,c:currentCls}); currentCls=(currentCls+1)%K; draw(); });
  kEl.addEventListener('input',()=>{kv.textContent=kEl.value; initW(); makeData(); draw();});
  lrEl.addEventListener('input',()=>{lrv.textContent=(+lrEl.value).toFixed(3);});
  bEl.addEventListener('input',()=>{bv.textContent=bEl.value;});
  document.getElementById('sm-step').onclick=()=>{ if(timer) return; timer=setInterval(trainStep,50); };
  document.getElementById('sm-pause').onclick=()=>{ clearInterval(timer); timer=null; };
  document.getElementById('sm-reset').onclick=()=>{ clearInterval(timer); timer=null; initW(); makeData(); draw(); };
  MOUNTED.sgd = () => { if(!W){ initW(); makeData(); } draw(); };
})();

/* ====================================================
   Activation functions
==================================================== */
(function(){
  const c=document.getElementById('act-c');
  const fns={
    sigmoid:{f:x=>1/(1+Math.exp(-x)), d:x=>{const s=1/(1+Math.exp(-x));return s*(1-s);}},
    tanh:{f:x=>Math.tanh(x), d:x=>1-Math.tanh(x)**2},
    relu:{f:x=>Math.max(0,x), d:x=>x>0?1:0},
    lrelu:{f:x=>x>0?x:0.1*x, d:x=>x>0?1:0.1},
    elu:{f:x=>x>0?x:Math.exp(x)-1, d:x=>x>0?1:Math.exp(x)},
    gelu:{f:x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x**3))), d:x=>{const e=1e-3;return ((0.5*(x+e)*(1+Math.tanh(Math.sqrt(2/Math.PI)*((x+e)+0.044715*(x+e)**3))))-(0.5*(x-e)*(1+Math.tanh(Math.sqrt(2/Math.PI)*((x-e)+0.044715*(x-e)**3)))))/(2*e);}},
    swish:{f:x=>x/(1+Math.exp(-x)), d:x=>{const s=1/(1+Math.exp(-x));return s+x*s*(1-s);}}
  };
  let cur='relu';
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    const x0=w/2,y0=h/2, sx=w/8, sy=h/4;
    g.strokeStyle='#2a3a63'; g.beginPath(); g.moveTo(0,y0);g.lineTo(w,y0); g.moveTo(x0,0);g.lineTo(x0,h); g.stroke();
    g.fillStyle='#8b96a8'; g.font='12px monospace'; g.fillText(cur, 10, 16);
    function plot(fn,col){
      g.strokeStyle=col; g.lineWidth=2; g.beginPath();
      for(let px=0;px<w;px++){ const x=(px-x0)/sx; const y=fn(x); const py=y0-y*sy; if(px===0) g.moveTo(px,py); else g.lineTo(px,py); }
      g.stroke();
    }
    plot(fns[cur].f,'#7cc4ff'); plot(fns[cur].d,'#ff7c9c');
  }
  document.querySelectorAll('[data-act]').forEach(b=>b.addEventListener('click',()=>{cur=b.dataset.act; draw();}));
  MOUNTED.act = draw;
})();

/* ====================================================
   MLP + Backprop (manual)
==================================================== */
(function(){
  const c=document.getElementById('mlp-c');
  const archSel=document.getElementById('mlp-arch'),actSel=document.getElementById('mlp-act');
  const lrEl=document.getElementById('mlp-lr'),lrv=document.getElementById('mlp-lrv');
  const dataSel=document.getElementById('mlp-data');
  const lossEl=document.getElementById('mlp-loss'),epEl=document.getElementById('mlp-ep');
  let net=null, pts=[], timer=null, epoch=0;
  function actFn(name){
    if(name==='relu') return {f:x=>Math.max(0,x), d:y=>y>0?1:0};
    if(name==='sigmoid') return {f:x=>1/(1+Math.exp(-x)), d:y=>y*(1-y)};
    return {f:x=>Math.tanh(x), d:y=>1-y*y};
  }
  function makeNet(){
    const hidden = archSel.value.split(',').map(s=>+s);
    const layers=[2,...hidden,1];
    const W=[],b=[];
    for(let i=0;i<layers.length-1;i++){
      const Wi=Array.from({length:layers[i+1]},()=>{
        const row=[]; for(let j=0;j<layers[i];j++) row.push(randn()*Math.sqrt(2/layers[i])); return row;
      });
      W.push(Wi); b.push(new Array(layers[i+1]).fill(0));
    }
    return {W,b,act:actSel.value,layers};
  }
  function forward(net,x){
    let h=x.slice(); const cache=[h];
    const A=actFn(net.act);
    for(let l=0;l<net.W.length;l++){
      const Wl=net.W[l], bl=net.b[l]; const next=[];
      for(let i=0;i<Wl.length;i++){
        let s=bl[i]; for(let j=0;j<h.length;j++) s+=Wl[i][j]*h[j];
        if(l===net.W.length-1) next.push(1/(1+Math.exp(-s))); // output sigmoid
        else next.push(A.f(s));
      }
      h=next; cache.push(h);
    }
    return cache;
  }
  function backprop(net,x,y,lr){
    const A=actFn(net.act);
    const cache=forward(net,x);
    const out=cache[cache.length-1][0];
    const loss = -(y*Math.log(out+1e-9)+(1-y)*Math.log(1-out+1e-9));
    let dh=[out-y]; // dL/dz for output (sigmoid+BCE)
    for(let l=net.W.length-1;l>=0;l--){
      const inp=cache[l], W=net.W[l], b=net.b[l];
      const dW=W.map(()=>new Array(inp.length).fill(0));
      const db=new Array(W.length).fill(0);
      for(let i=0;i<W.length;i++){
        db[i]=dh[i];
        for(let j=0;j<inp.length;j++) dW[i][j]=dh[i]*inp[j];
      }
      // grad w.r.t input
      let dnext=new Array(inp.length).fill(0);
      for(let j=0;j<inp.length;j++){
        let s=0; for(let i=0;i<W.length;i++) s+=W[i][j]*dh[i];
        dnext[j]=s;
      }
      // apply activation deriv for layer below (which is inp)
      if(l>0){
        for(let j=0;j<inp.length;j++) dnext[j]*=A.d(inp[j]);
      }
      // update
      for(let i=0;i<W.length;i++){
        b[i]-=lr*db[i];
        for(let j=0;j<inp.length;j++) W[i][j]-=lr*dW[i][j];
      }
      dh=dnext;
    }
    return loss;
  }
  function makeData(){
    pts=[]; const ww=c.clientWidth, hh=c.clientHeight||420;
    const cx=ww/2, cy=hh/2, R=Math.min(ww,hh)*.35;
    const ds=dataSel.value;
    if(ds==='moons'){
      for(let i=0;i<60;i++){ const t=Math.PI*rng(); pts.push({x:cx-R*.4+R*Math.cos(t), y:cy+R*Math.sin(t)+randn()*8, c:0}); }
      for(let i=0;i<60;i++){ const t=Math.PI*rng(); pts.push({x:cx+R*.4-R*Math.cos(t), y:cy-R*Math.sin(t)+randn()*8, c:1}); }
    } else if(ds==='xor'){
      for(let i=0;i<40;i++){ pts.push({x:cx-R*.5+randn()*R*.15, y:cy-R*.5+randn()*R*.15, c:0}); pts.push({x:cx+R*.5+randn()*R*.15, y:cy+R*.5+randn()*R*.15, c:0}); pts.push({x:cx+R*.5+randn()*R*.15, y:cy-R*.5+randn()*R*.15, c:1}); pts.push({x:cx-R*.5+randn()*R*.15, y:cy+R*.5+randn()*R*.15, c:1}); }
    } else if(ds==='circles'){
      for(let i=0;i<60;i++){ const t=2*Math.PI*rng(); pts.push({x:cx+R*.3*Math.cos(t)+randn()*5, y:cy+R*.3*Math.sin(t)+randn()*5, c:0}); }
      for(let i=0;i<60;i++){ const t=2*Math.PI*rng(); pts.push({x:cx+R*.9*Math.cos(t)+randn()*5, y:cy+R*.9*Math.sin(t)+randn()*5, c:1}); }
    } else { // spirals
      for(let i=0;i<70;i++){ const r=i/70, t=r*5+0; pts.push({x:cx+r*R*Math.cos(t)+randn()*4, y:cy+r*R*Math.sin(t)+randn()*4, c:0}); }
      for(let i=0;i<70;i++){ const r=i/70, t=r*5+Math.PI; pts.push({x:cx+r*R*Math.cos(t)+randn()*4, y:cy+r*R*Math.sin(t)+randn()*4, c:1}); }
    }
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(net){
      const step=10;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        const xn=(x/w)*2-1, yn=(y/h)*2-1;
        const out=forward(net,[xn,yn]);
        const p=out[out.length-1][0];
        g.fillStyle = `rgba(${Math.floor(255-p*131)},${Math.floor(124+p*0)},${Math.floor(156+p*99)},.22)`;
        // Simpler color: lerp between two class colors
        g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.4})` : `rgba(124,196,255,${.10+(.5-p)*.4})`;
        g.fillRect(x,y,step,step);
      }
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`arch=${net?net.layers.join('→'):'-'}  act=${actSel.value}`, 10, 16);
  }
  function trainTick(){
    if(!net||!pts.length) return;
    let totalLoss=0; const lr=+lrEl.value, ww=c.clientWidth, hh=c.clientHeight||420;
    for(let i=0;i<60;i++){
      const p=pts[Math.floor(rng()*pts.length)];
      const x=[(p.x/ww)*2-1,(p.y/hh)*2-1];
      totalLoss += backprop(net, x, p.c, lr);
    }
    epoch++; epEl.textContent=epoch; lossEl.textContent=(totalLoss/60).toFixed(3);
    draw();
  }
  function reset(){ clearInterval(timer); timer=null; epoch=0; epEl.textContent=0; net=makeNet(); draw(); }
  [archSel,actSel].forEach(s=>s.addEventListener('change',reset));
  dataSel.addEventListener('change',()=>{ makeData(); reset(); });
  lrEl.addEventListener('input',()=>{lrv.textContent=(+lrEl.value).toFixed(3);});
  document.getElementById('mlp-run').onclick=()=>{ if(!timer) timer=setInterval(trainTick,30); };
  document.getElementById('mlp-pause').onclick=()=>{ clearInterval(timer); timer=null; };
  document.getElementById('mlp-reset').onclick=()=>{ reset(); };
  MOUNTED.mlp = () => { if(!pts.length){ makeData(); } if(!net){ net=makeNet(); } draw(); };
})();

/* ====================================================
   Decision Tree (classification)
==================================================== */
function buildTree(pts, depth, maxDepth, minLeaf, crit){
  if(pts.length===0) return {leaf:true, pred:0};
  let c0=0,c1=0; pts.forEach(p=>{ if(p.c===0) c0++; else c1++; });
  const pred = c1>=c0?1:0;
  function impurity(n0,n1){
    const n=n0+n1; if(!n) return 0;
    const p0=n0/n, p1=n1/n;
    if(crit==='ent'){ let h=0; if(p0>0) h-=p0*Math.log2(p0); if(p1>0) h-=p1*Math.log2(p1); return h; }
    return 1 - p0*p0 - p1*p1; // gini
  }
  if(depth>=maxDepth || pts.length<=minLeaf || c0===0 || c1===0) return {leaf:true, pred, n0:c0, n1:c1};
  let best=null;
  for(const feat of ['x','y']){
    const sorted=pts.slice().sort((a,b)=>a[feat]-b[feat]);
    for(let i=1;i<sorted.length;i++){
      if(sorted[i][feat]===sorted[i-1][feat]) continue;
      const thr=(sorted[i][feat]+sorted[i-1][feat])/2;
      let l0=0,l1=0,r0=0,r1=0;
      for(const p of sorted){ if(p[feat]<=thr){ if(p.c===0) l0++; else l1++; } else { if(p.c===0) r0++; else r1++; } }
      const nl=l0+l1, nr=r0+r1; if(nl<minLeaf||nr<minLeaf) continue;
      const imp = (nl*impurity(l0,l1) + nr*impurity(r0,r1)) / pts.length;
      if(!best || imp<best.imp) best={imp,feat,thr};
    }
  }
  if(!best) return {leaf:true, pred, n0:c0, n1:c1};
  const left=pts.filter(p=>p[best.feat]<=best.thr);
  const right=pts.filter(p=>p[best.feat]>best.thr);
  return {leaf:false, feat:best.feat, thr:best.thr,
    left:buildTree(left,depth+1,maxDepth,minLeaf,crit),
    right:buildTree(right,depth+1,maxDepth,minLeaf,crit), n0:c0,n1:c1};
}
function predictTree(t,p){ while(!t.leaf){ t=p[t.feat]<=t.thr?t.left:t.right; } return t.pred; }

(function(){
  const c=document.getElementById('dt-c'), ct=document.getElementById('dt-tree');
  const dEl=document.getElementById('dt-d'),dv=document.getElementById('dt-dv');
  const mEl=document.getElementById('dt-m'),mv=document.getElementById('dt-mv');
  const critEl=document.getElementById('dt-crit'),clsEl=document.getElementById('dt-cls');
  let pts=[];
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    let tree=null;
    if(pts.length){
      tree=buildTree(pts,0,+dEl.value,+mEl.value,critEl.value);
      const step=8;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        const p=predictTree(tree,{x,y});
        g.fillStyle=classColorAlpha(p,.2); g.fillRect(x,y,step,step);
      }
      // splits
      function drawSplits(t, xmin,xmax,ymin,ymax){
        if(t.leaf) return;
        g.strokeStyle='#fff'; g.lineWidth=1.2;
        if(t.feat==='x'){ g.beginPath(); g.moveTo(t.thr,ymin); g.lineTo(t.thr,ymax); g.stroke();
          drawSplits(t.left,xmin,t.thr,ymin,ymax); drawSplits(t.right,t.thr,xmax,ymin,ymax);
        } else { g.beginPath(); g.moveTo(xmin,t.thr); g.lineTo(xmax,t.thr); g.stroke();
          drawSplits(t.left,xmin,xmax,ymin,t.thr); drawSplits(t.right,xmin,xmax,t.thr,ymax); }
      }
      drawSplits(tree,0,w,0,h);
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
    // Tree panel
    const {g:gt, w:wt, h:ht} = fitCanvas(ct); gridBg(gt,wt,ht);
    if(tree) drawTreeNode(gt, tree, wt/2, 24, wt/3, 50);
  }
  function drawTreeNode(g, t, x, y, span, dy){
    if(t.leaf){
      g.fillStyle = classColorAlpha(t.pred,.4);
      g.fillRect(x-22,y-12,44,24);
      g.strokeStyle = classColor(t.pred); g.strokeRect(x-22,y-12,44,24);
      g.fillStyle='#e6edf3'; g.font='11px monospace'; g.textAlign='center';
      g.fillText(`y=${t.pred}`,x,y+4); g.textAlign='start';
      return;
    }
    g.fillStyle='#172238'; g.strokeStyle='#7cc4ff'; g.fillRect(x-30,y-12,60,24); g.strokeRect(x-30,y-12,60,24);
    g.fillStyle='#cfe3ff'; g.font='10px monospace'; g.textAlign='center';
    g.fillText(`${t.feat}≤${t.thr.toFixed(0)}`, x, y+4); g.textAlign='start';
    const xl=x-span, xr=x+span, ny=y+dy;
    if(ny<g.canvas.height/(window.devicePixelRatio||1)-15){
      g.strokeStyle='#243047'; g.beginPath(); g.moveTo(x,y+12); g.lineTo(xl,ny-12); g.moveTo(x,y+12); g.lineTo(xr,ny-12); g.stroke();
      drawTreeNode(g, t.left, xl, ny, span/2, dy);
      drawTreeNode(g, t.right, xr, ny, span/2, dy);
    }
  }
  bindClick(c,(x,y)=>{ pts.push({x,y,c:+clsEl.value}); draw(); });
  [dEl,mEl,critEl].forEach(el=>el.addEventListener('input',()=>{ dv.textContent=dEl.value; mv.textContent=mEl.value; draw(); }));
  document.getElementById('dt-demo').onclick=()=>{
    pts=[]; const W=c.clientWidth,H=c.clientHeight||400;
    for(let i=0;i<40;i++){ const r=Math.random()*60+40, t=Math.random()*2*Math.PI; pts.push({x:W/2+r*Math.cos(t), y:H/2+r*Math.sin(t), c:0}); }
    for(let i=0;i<40;i++){ const r=Math.random()*40, t=Math.random()*2*Math.PI; pts.push({x:W/2+r*Math.cos(t), y:H/2+r*Math.sin(t), c:1}); }
    draw();
  };
  document.getElementById('dt-clear').onclick=()=>{pts=[];draw();};
  MOUNTED.dtree = () => { if(!pts.length) document.getElementById('dt-demo').onclick(); else draw(); };
})();

/* ====================================================
   Bagging (and Random Forest below)
==================================================== */
function genTwoClusters(W,H, sep=80){
  const pts=[];
  for(let i=0;i<40;i++) pts.push({x:W*.35+randn()*sep*.7, y:H*.55+randn()*sep*.7, c:0});
  for(let i=0;i<40;i++) pts.push({x:W*.65+randn()*sep*.7, y:H*.45+randn()*sep*.7, c:1});
  // mix
  for(let i=0;i<10;i++) pts.push({x:W*.5+randn()*60, y:H*.5+randn()*60, c:Math.round(rng())});
  return pts;
}
// Bootstrap replicate via the tested ensemble core in ./src/ensemble.js
// (shares the page rng so the visualisation is unchanged).
function bootstrap(pts){ return bagBootstrap(pts, rng); }
function buildTreeRF(pts, depth, maxDepth, mtry){
  if(pts.length===0) return {leaf:true, pred:0};
  let c0=0,c1=0; pts.forEach(p=>{ if(p.c===0) c0++; else c1++; });
  const pred = c1>=c0?1:0;
  if(depth>=maxDepth || c0===0 || c1===0) return {leaf:true,pred};
  const feats=['x','y']; const chosen = (mtry>=2)?feats: [feats[Math.floor(rng()*2)]];
  let best=null;
  for(const feat of chosen){
    const sorted=pts.slice().sort((a,b)=>a[feat]-b[feat]);
    for(let i=1;i<sorted.length;i++){
      if(sorted[i][feat]===sorted[i-1][feat]) continue;
      const thr=(sorted[i][feat]+sorted[i-1][feat])/2;
      let l0=0,l1=0,r0=0,r1=0;
      for(const p of sorted){ if(p[feat]<=thr){ if(p.c===0) l0++; else l1++; } else { if(p.c===0) r0++; else r1++; } }
      const nl=l0+l1, nr=r0+r1; if(nl<1||nr<1) continue;
      const gini=(nl*(1-(l0/nl)**2-(l1/nl)**2) + nr*(1-(r0/nr)**2-(r1/nr)**2))/pts.length;
      if(!best||gini<best.imp) best={imp:gini,feat,thr};
    }
  }
  if(!best) return {leaf:true,pred};
  return {leaf:false,feat:best.feat,thr:best.thr,
    left:buildTreeRF(pts.filter(p=>p[best.feat]<=best.thr),depth+1,maxDepth,mtry),
    right:buildTreeRF(pts.filter(p=>p[best.feat]>best.thr),depth+1,maxDepth,mtry)};
}

(function(){
  const c=document.getElementById('bag-c');
  const tEl=document.getElementById('bag-t'),tv=document.getElementById('bag-tv');
  const dEl=document.getElementById('bag-d'),dv=document.getElementById('bag-dv');
  const shwEl=document.getElementById('bag-show');
  let pts=null, trees=[];
  function refit(){
    if(!pts) return;
    trees=[];
    const T=+tEl.value, depth=+dEl.value;
    for(let i=0;i<T;i++) trees.push(buildTreeRF(bootstrap(pts), 0, depth, 2));
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(!pts){ pts=genTwoClusters(w,h); refit(); }
    const step=8;
    for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
      let v=0; for(const t of trees) v+=predictTree(t,{x,y});
      const p=v/trees.length;
      if(shwEl.value==='avg'){
        g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.4})` : `rgba(124,196,255,${.10+(.5-p)*.4})`;
        g.fillRect(x,y,step,step);
      } else {
        // accumulate alpha from each tree
        trees.slice(0,Math.min(20,trees.length)).forEach(t=>{
          const cls=predictTree(t,{x,y});
          g.fillStyle=classColorAlpha(cls,.04); g.fillRect(x,y,step,step);
        });
      }
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
  }
  [tEl,dEl].forEach(el=>el.addEventListener('input',()=>{tv.textContent=tEl.value; dv.textContent=dEl.value; refit(); draw();}));
  shwEl.addEventListener('change',draw);
  document.getElementById('bag-demo').onclick=()=>{ pts=genTwoClusters(c.clientWidth,c.clientHeight||400); refit(); draw(); };
  MOUNTED.bagging = () => { if(!pts) draw(); else draw(); };
})();

(function(){
  const c=document.getElementById('rf-c');
  const tEl=document.getElementById('rf-t'),tv=document.getElementById('rf-tv');
  const dEl=document.getElementById('rf-d'),dv=document.getElementById('rf-dv');
  const mtryEl=document.getElementById('rf-mtry');
  let pts=null, trees=[];
  function refit(){
    if(!pts) return;
    trees=[]; const T=+tEl.value, depth=+dEl.value, mtry=+mtryEl.value;
    for(let i=0;i<T;i++) trees.push(buildTreeRF(bootstrap(pts),0,depth,mtry));
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(!pts){ pts=genTwoClusters(w,h); refit(); }
    const step=8;
    for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
      let v=0; for(const t of trees) v+=predictTree(t,{x,y});
      const p=v/trees.length;
      g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.4})` : `rgba(124,196,255,${.10+(.5-p)*.4})`;
      g.fillRect(x,y,step,step);
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
  }
  [tEl,dEl,mtryEl].forEach(el=>el.addEventListener('input',()=>{tv.textContent=tEl.value; dv.textContent=dEl.value; refit(); draw();}));
  document.getElementById('rf-demo').onclick=()=>{ pts=genTwoClusters(c.clientWidth,c.clientHeight||400); refit(); draw(); };
  MOUNTED.rf = () => { if(!pts) draw(); else draw(); };
})();

/* ====================================================
   AdaBoost (decision stumps)
==================================================== */
(function(){
  const c=document.getElementById('ab-c');
  const tEl=document.getElementById('ab-t'),tv=document.getElementById('ab-tv');
  const errEl=document.getElementById('ab-err');
  let pts=null, stumps=[], W=[], timer=null;
  function genData(){
    pts=[]; const ww=c.clientWidth, hh=c.clientHeight||400;
    for(let i=0;i<35;i++){ pts.push({x:ww*.35+randn()*40, y:hh*.5+randn()*40, c:-1}); }
    for(let i=0;i<35;i++){ pts.push({x:ww*.65+randn()*40, y:hh*.5+randn()*40, c:1}); }
    // a few mixers
    for(let i=0;i<6;i++){ pts.push({x:ww*.5+randn()*70, y:hh*.5+randn()*70, c: rng()<.5?-1:1}); }
    W = new Array(pts.length).fill(1/pts.length);
    stumps=[];
  }
  function findStump(){
    let best={err:Infinity};
    for(const feat of ['x','y']){
      const vals = pts.map(p=>p[feat]).sort((a,b)=>a-b);
      for(let i=0;i<vals.length-1;i++){
        const thr=(vals[i]+vals[i+1])/2;
        for(const sign of [1,-1]){
          let err=0;
          pts.forEach((p,j)=>{ const pred = (p[feat]<=thr?1:-1)*sign; if(pred!==p.c) err+=W[j]; });
          if(err<best.err) best={err,feat,thr,sign};
        }
      }
    }
    return best;
  }
  function addRound(){
    const s=findStump();
    if(s.err>=0.5){ return; }
    const alpha = 0.5*Math.log((1-s.err)/Math.max(s.err,1e-9));
    // update weights
    let Z=0;
    pts.forEach((p,j)=>{
      const pred=(p[s.feat]<=s.thr?1:-1)*s.sign;
      W[j]*=Math.exp(-alpha*p.c*pred); Z+=W[j];
    });
    for(let j=0;j<W.length;j++) W[j]/=Z;
    stumps.push({...s,alpha});
  }
  function predictAB(p){
    let s=0; stumps.forEach(st => { s += st.alpha * ((p[st.feat]<=st.thr?1:-1)*st.sign); });
    return s;
  }
  function trainError(){
    let err=0; pts.forEach(p=>{ if(Math.sign(predictAB(p))!==p.c) err++; }); return err/pts.length;
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(!pts) genData();
    // boundary
    const step=8;
    for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
      const s=predictAB({x,y});
      const p = 1/(1+Math.exp(-s*2));
      g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.5})` : `rgba(124,196,255,${.10+(.5-p)*.5})`;
      g.fillRect(x,y,step,step);
    }
    // stumps lines
    stumps.forEach((s,i)=>{
      g.strokeStyle = `rgba(255,255,255,${.05+.6*Math.abs(s.alpha)/Math.max(...stumps.map(x=>Math.abs(x.alpha)))})`;
      g.lineWidth=1; g.beginPath();
      if(s.feat==='x'){ g.moveTo(s.thr,0); g.lineTo(s.thr,h); } else { g.moveTo(0,s.thr); g.lineTo(w,s.thr); }
      g.stroke();
    });
    // points sized by weight
    const maxW=Math.max(...W);
    pts.forEach((p,j)=>{ const r=3 + 12*W[j]/maxW; drawPoint(g,p.x,p.y,p.c, r); });
    errEl.textContent = stumps.length? (trainError()*100).toFixed(1)+'%' : '—';
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`T=${stumps.length}`,10,16);
  }
  function setRound(T){
    if(!pts) genData();
    if(stumps.length>T){ // rebuild
      W=new Array(pts.length).fill(1/pts.length); stumps=[];
    }
    while(stumps.length<T) addRound();
    draw();
  }
  tEl.addEventListener('input',()=>{tv.textContent=tEl.value; setRound(+tEl.value);});
  document.getElementById('ab-play').onclick=()=>{
    if(timer) return;
    timer=setInterval(()=>{
      let t=+tEl.value; t=Math.min(60,t+1); tEl.value=t; tv.textContent=t; setRound(t);
      if(t>=60){ clearInterval(timer); timer=null; }
    },150);
  };
  document.getElementById('ab-pause').onclick=()=>{clearInterval(timer); timer=null;};
  document.getElementById('ab-demo').onclick=()=>{ genData(); tEl.value=1; tv.textContent=1; setRound(1); };
  MOUNTED.boost = () => { if(!pts){ genData(); setRound(+tEl.value); } else draw(); };
})();

/* ====================================================
   Kernel trick — 2D and 3D paraboloid lift
==================================================== */
(function(){
  const c2=document.getElementById('k2d'), c3=document.getElementById('k3d');
  const rEl=document.getElementById('k-r'),rv=document.getElementById('k-rv');
  const aEl=document.getElementById('k-a'),av=document.getElementById('k-av');
  let pts=[];
  function gen(){
    pts=[]; const w=c2.clientWidth, h=c2.clientHeight||360, cx=w/2, cy=h/2;
    for(let i=0;i<40;i++){ const t=2*Math.PI*rng(), r=(40+randn()*8); pts.push({x:cx+r*Math.cos(t), y:cy+r*Math.sin(t), c:0}); }
    for(let i=0;i<40;i++){ const t=2*Math.PI*rng(), r=(100+randn()*8); pts.push({x:cx+r*Math.cos(t), y:cy+r*Math.sin(t), c:1}); }
    draw();
  }
  function draw(){
    const {g:g2,w:w2,h:h2}=fitCanvas(c2); gridBg(g2,w2,h2);
    const {g:g3,w:w3,h:h3}=fitCanvas(c3); gridBg(g3,w3,h3);
    // 2D
    pts.forEach(p=>drawPoint(g2,p.x,p.y,p.c,5));
    // separator circle (kernel boundary candidate)
    g2.strokeStyle='#7cffb2'; g2.setLineDash([4,3]); g2.beginPath(); g2.arc(w2/2,h2/2,70,0,Math.PI*2); g2.stroke(); g2.setLineDash([]);
    // 3D paraboloid
    const cx=w2/2, cy=h2/2;
    const theta=+rEl.value*Math.PI/180;
    const alpha=+aEl.value;
    // basis
    function proj(X,Y,Z){
      // simple axonometric: rotate around X
      const cosA=Math.cos(theta), sinA=Math.sin(theta);
      const y2 = Y*cosA - Z*sinA, z2 = Y*sinA + Z*cosA;
      return {sx: w3/2 + X*0.45, sy: h3*.7 - y2*0.45 - z2*0.0};
    }
    // axes
    g3.strokeStyle='#2a3a63'; g3.lineWidth=1;
    [[-200,200,0,0,0,0],[0,0,-200,200,0,0],[0,0,0,0,0,200]].forEach(a=>{
      const p1=proj(a[0],a[2],a[4]), p2=proj(a[1],a[3],a[5]);
      g3.beginPath(); g3.moveTo(p1.sx,p1.sy); g3.lineTo(p2.sx,p2.sy); g3.stroke();
    });
    // paraboloid grid
    g3.strokeStyle='rgba(255,255,255,.06)'; g3.lineWidth=1;
    for(let i=-160;i<=160;i+=20){
      g3.beginPath();
      for(let j=-160;j<=160;j+=5){
        const Z=(i*i+j*j)*alpha/200;
        const p=proj(i,j,Z);
        if(j===-160) g3.moveTo(p.sx,p.sy); else g3.lineTo(p.sx,p.sy);
      }
      g3.stroke();
      g3.beginPath();
      for(let j=-160;j<=160;j+=5){
        const Z=(j*j+i*i)*alpha/200;
        const p=proj(j,i,Z);
        if(j===-160) g3.moveTo(p.sx,p.sy); else g3.lineTo(p.sx,p.sy);
      }
      g3.stroke();
    }
    // separating plane at Z = z0 (sphere radius 70 in 2D => x²+y² = 4900 ⇒ Z=4900*alpha/200 = 24.5α)
    const z0 = (70*70)*alpha/200;
    g3.strokeStyle='rgba(124,255,178,.6)'; g3.lineWidth=1;
    g3.beginPath();
    for(let i=-180;i<=180;i+=10){
      const p=proj(i,-180,z0); const q=proj(i,180,z0);
      g3.moveTo(p.sx,p.sy); g3.lineTo(q.sx,q.sy);
    }
    g3.stroke();
    // points lifted
    pts.forEach(p=>{
      const x=p.x-cx, y=p.y-cy, Z=(x*x+y*y)*alpha/200;
      const sp=proj(x,y,Z);
      drawPoint(g3, sp.sx, sp.sy, p.c, 5);
    });
    g3.fillStyle='#8b96a8'; g3.font='12px monospace';
    g3.fillText(`φ(x,y) = (x, y, α(x²+y²))`, 10, 16);
  }
  rEl.addEventListener('input',()=>{rv.textContent=rEl.value+'°'; draw();});
  aEl.addEventListener('input',()=>{av.textContent=(+aEl.value).toFixed(2); draw();});
  document.getElementById('k-demo').onclick=gen;
  MOUNTED.kernel = () => { if(!pts.length) gen(); else draw(); };
})();

/* ====================================================
   SVM — soft margin, kernel via SMO-lite
==================================================== */
(function(){
  const c=document.getElementById('svm-c');
  const cEl=document.getElementById('svm-c1'),cv=document.getElementById('svm-cv');
  const kEl=document.getElementById('svm-k');
  const gEl=document.getElementById('svm-g'),gv=document.getElementById('svm-gv');
  const clsEl=document.getElementById('svm-cls');
  let pts=[]; let model=null;
  function K(a,b){
    const dx=a.x-b.x, dy=a.y-b.y;
    if(kEl.value==='rbf'){
      const g=Math.pow(10,+gEl.value);
      return Math.exp(-g*(dx*dx+dy*dy)/10000);
    } else if(kEl.value==='poly'){
      const W=c.clientWidth||500, H=c.clientHeight||420;
      const ax=(a.x-W/2)/W*4, ay=(a.y-H/2)/H*4;
      const bx=(b.x-W/2)/W*4, by=(b.y-H/2)/H*4;
      return Math.pow(ax*bx+ay*by+1,3);
    } else {
      const W=c.clientWidth||500, H=c.clientHeight||420;
      const ax=(a.x-W/2)/W*4, ay=(a.y-H/2)/H*4;
      const bx=(b.x-W/2)/W*4, by=(b.y-H/2)/H*4;
      return ax*bx+ay*by;
    }
  }
  function trainSMO(){
    const n=pts.length; if(!n) return;
    const C=Math.pow(10,+cEl.value);
    const alpha=new Array(n).fill(0); let bias=0;
    const tol=1e-3; const maxPasses=8;
    function f(x){
      let s=bias;
      for(let i=0;i<n;i++) s+=alpha[i]*pts[i].c*K(pts[i],x);
      return s;
    }
    // Precompute kernel?
    let passes=0;
    while(passes<maxPasses){
      let changed=0;
      for(let i=0;i<n;i++){
        const Ei=f(pts[i])-pts[i].c;
        if((pts[i].c*Ei<-tol&&alpha[i]<C)||(pts[i].c*Ei>tol&&alpha[i]>0)){
          let j=Math.floor(rng()*n); if(j===i) j=(j+1)%n;
          const Ej=f(pts[j])-pts[j].c;
          const aiOld=alpha[i], ajOld=alpha[j];
          let L,H2;
          if(pts[i].c!==pts[j].c){ L=Math.max(0,alpha[j]-alpha[i]); H2=Math.min(C,C+alpha[j]-alpha[i]); }
          else { L=Math.max(0,alpha[i]+alpha[j]-C); H2=Math.min(C,alpha[i]+alpha[j]); }
          if(L===H2) continue;
          const eta = 2*K(pts[i],pts[j])-K(pts[i],pts[i])-K(pts[j],pts[j]);
          if(eta>=0) continue;
          let ajNew = alpha[j] - pts[j].c*(Ei-Ej)/eta;
          ajNew = clamp(ajNew, L, H2);
          if(Math.abs(ajNew-ajOld)<1e-5) continue;
          alpha[j]=ajNew;
          alpha[i] += pts[i].c*pts[j].c*(ajOld-ajNew);
          const b1 = bias - Ei - pts[i].c*(alpha[i]-aiOld)*K(pts[i],pts[i]) - pts[j].c*(alpha[j]-ajOld)*K(pts[i],pts[j]);
          const b2 = bias - Ej - pts[i].c*(alpha[i]-aiOld)*K(pts[i],pts[j]) - pts[j].c*(alpha[j]-ajOld)*K(pts[j],pts[j]);
          if(alpha[i]>0&&alpha[i]<C) bias=b1;
          else if(alpha[j]>0&&alpha[j]<C) bias=b2;
          else bias=(b1+b2)/2;
          changed++;
        }
      }
      if(!changed) passes++; else passes=0;
      if(passes>maxPasses) break;
    }
    model={alpha,bias,f};
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(pts.length){
      trainSMO();
      const step=8;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        const s=model.f({x,y});
        const p=1/(1+Math.exp(-s*2));
        g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.5})` : `rgba(124,196,255,${.10+(.5-p)*.5})`;
        g.fillRect(x,y,step,step);
      }
      // boundary level set
      g.strokeStyle='#fff'; g.lineWidth=1;
      // crude contour via marching
      for(let x=0;x<w;x+=4) for(let y=0;y<h;y+=4){
        const v=model.f({x,y});
        if(Math.abs(v)<0.05){ g.fillStyle='rgba(255,255,255,.9)'; g.fillRect(x,y,2,2); }
        if(Math.abs(v-1)<0.05||Math.abs(v+1)<0.05){ g.fillStyle='rgba(255,255,255,.4)'; g.fillRect(x,y,2,2); }
      }
    }
    pts.forEach((p,i)=>{
      drawPoint(g,p.x,p.y,p.c===1?1:0,5);
      if(model && model.alpha[i]>1e-3){
        g.strokeStyle='#ffb86b'; g.lineWidth=2; g.beginPath(); g.arc(p.x,p.y,9,0,Math.PI*2); g.stroke();
      }
    });
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`kernel=${kEl.value}  C=10^${(+cEl.value).toFixed(1)}`,10,16);
  }
  bindClick(c,(x,y)=>{ pts.push({x,y,c:+clsEl.value}); draw(); });
  [cEl,kEl,gEl].forEach(el=>el.addEventListener('input',()=>{
    cv.textContent=Math.pow(10,+cEl.value).toFixed(2); gv.textContent=Math.pow(10,+gEl.value).toFixed(2); draw();
  }));
  document.getElementById('svm-demo').onclick=()=>{
    pts=[]; const W=c.clientWidth, H=c.clientHeight||420;
    for(let i=0;i<25;i++) pts.push({x:W*.3+randn()*35, y:H*.6+randn()*35, c:1});
    for(let i=0;i<25;i++) pts.push({x:W*.7+randn()*35, y:H*.4+randn()*35, c:-1});
    draw();
  };
  document.getElementById('svm-clear').onclick=()=>{pts=[];model=null;draw();};
  MOUNTED.svm = () => { if(!pts.length) document.getElementById('svm-demo').onclick(); else draw(); };
})();

/* ====================================================
   PCA
==================================================== */
(function(){
  const c=document.getElementById('pca-c'), cp=document.getElementById('pca-proj');
  const kEl=document.getElementById('pca-k'),varEl=document.getElementById('pca-var');
  let pts=[]; let drag=-1;
  function demo(){
    pts=[]; const w=c.clientWidth, h=c.clientHeight||360, cx=w/2, cy=h/2;
    for(let i=0;i<60;i++){ const r=randn()*60, t=randn()*15; const X=r,Y=t; const ang=Math.PI/4;
      pts.push({x:cx+X*Math.cos(ang)-Y*Math.sin(ang), y:cy+X*Math.sin(ang)+Y*Math.cos(ang)});
    }
    draw();
  }
  function eig2x2(a,b,c2,d){
    const tr=a+d, det=a*d-b*c2;
    const disc=Math.sqrt(Math.max(0,tr*tr/4 - det));
    const l1=tr/2+disc, l2=tr/2-disc;
    function vec(l){ if(Math.abs(b)>1e-9){const v=[b,l-a]; const n=Math.hypot(v[0],v[1]); return [v[0]/n,v[1]/n];}
      if(Math.abs(c2)>1e-9){const v=[l-d,c2]; const n=Math.hypot(v[0],v[1]); return [v[0]/n,v[1]/n];}
      return [1,0];
    }
    return {l1,l2,v1:vec(l1),v2:vec(l2)};
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    const {g:gp, w:wp, h:hp} = fitCanvas(cp); gridBg(gp,wp,hp);
    if(pts.length<2){ pts.forEach(p=>drawPoint(g,p.x,p.y,0,5)); return; }
    const cx=pts.reduce((s,p)=>s+p.x,0)/pts.length, cy=pts.reduce((s,p)=>s+p.y,0)/pts.length;
    let sxx=0,syy=0,sxy=0;
    pts.forEach(p=>{ const dx=p.x-cx, dy=p.y-cy; sxx+=dx*dx; syy+=dy*dy; sxy+=dx*dy; });
    sxx/=pts.length; syy/=pts.length; sxy/=pts.length;
    const E=eig2x2(sxx,sxy,sxy,syy);
    const v1=E.v1, v2=E.v2, l1=Math.max(0,E.l1), l2=Math.max(0,E.l2);
    pts.forEach(p=>drawPoint(g,p.x,p.y,0,5));
    // PC axes
    const s1=Math.sqrt(l1)*2, s2=Math.sqrt(l2)*2;
    g.strokeStyle='#ff7c9c'; g.lineWidth=2; g.beginPath(); g.moveTo(cx-v1[0]*s1, cy-v1[1]*s1); g.lineTo(cx+v1[0]*s1, cy+v1[1]*s1); g.stroke();
    g.strokeStyle='#7cffb2'; g.lineWidth=2; g.beginPath(); g.moveTo(cx-v2[0]*s2, cy-v2[1]*s2); g.lineTo(cx+v2[0]*s2, cy+v2[1]*s2); g.stroke();
    g.fillStyle='#ff7c9c'; g.font='11px monospace'; g.fillText('PC1', cx+v1[0]*s1+4, cy+v1[1]*s1);
    g.fillStyle='#7cffb2'; g.fillText('PC2', cx+v2[0]*s2+4, cy+v2[1]*s2);
    // variance explained
    const tot=l1+l2, pc1=l1/tot, pc2=l2/tot;
    varEl.textContent = `PC1 ${(pc1*100).toFixed(1)}%  PC2 ${(pc2*100).toFixed(1)}%`;
    // Projection canvas
    const k=+kEl.value;
    const cxp=wp/2, cyp=hp/2;
    pts.forEach(p=>{
      const dx=p.x-cx, dy=p.y-cy;
      const a=dx*v1[0]+dy*v1[1], b=dx*v2[0]+dy*v2[1];
      let px, py;
      if(k===1){ px = cxp + a; py = cyp; }
      else { px=cxp+a; py=cyp+b; }
      drawPoint(gp,px,py,0,5);
    });
    gp.strokeStyle='#2a3a63'; gp.beginPath(); gp.moveTo(20,cyp); gp.lineTo(wp-20,cyp); gp.stroke();
    if(k===2){ gp.beginPath(); gp.moveTo(cxp,20); gp.lineTo(cxp,hp-20); gp.stroke(); }
    gp.fillStyle='#8b96a8'; gp.font='11px monospace'; gp.fillText('PC1', wp-30, cyp-4); if(k===2) gp.fillText('PC2', cxp+4, 14);
  }
  c.addEventListener('mousedown',e=>{
    const r=c.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
    let h=-1; pts.forEach((p,i)=>{ if(Math.hypot(p.x-x,p.y-y)<8) h=i; });
    if(h<0){ pts.push({x,y}); draw(); } else drag=h;
  });
  window.addEventListener('mousemove',e=>{ if(drag<0) return; const r=c.getBoundingClientRect(); pts[drag].x=e.clientX-r.left; pts[drag].y=e.clientY-r.top; draw(); });
  window.addEventListener('mouseup',()=>drag=-1);
  kEl.addEventListener('change',draw);
  document.getElementById('pca-demo').onclick=demo;
  document.getElementById('pca-clear').onclick=()=>{pts=[];draw();};
  MOUNTED.pca = () => { if(!pts.length) demo(); else draw(); };
})();

/* ====================================================
   MDS (SMACOF-style)
==================================================== */
(function(){
  const c=document.getElementById('mds-c');
  const dataSel=document.getElementById('mds-data');
  const stressEl=document.getElementById('mds-stress');
  let D=null, pos=null, names=null, timer=null;
  function build(){
    const ds=dataSel.value;
    if(ds==='cities'){
      // mock 'distances' between fictional cities
      names=['α','β','γ','δ','ε','ζ','η','θ'];
      D=Array.from({length:8},()=>new Array(8).fill(0));
      // 3D positions
      const seed=[[0,0,0],[3,0,0],[3,3,0],[0,3,0],[1.5,1.5,2],[1.5,1.5,-2],[0,0,3],[3,3,3]];
      for(let i=0;i<8;i++) for(let j=0;j<8;j++) D[i][j]=Math.hypot(seed[i][0]-seed[j][0],seed[i][1]-seed[j][1],seed[i][2]-seed[j][2])*60;
    } else if(ds==='grid'){
      const N=16; names=null; D=Array.from({length:N},()=>new Array(N));
      const pos3=[]; for(let i=0;i<N;i++) pos3.push([rng()*4-2,rng()*4-2,rng()*4-2]);
      for(let i=0;i<N;i++) for(let j=0;j<N;j++) D[i][j]=Math.hypot(pos3[i][0]-pos3[j][0],pos3[i][1]-pos3[j][1],pos3[i][2]-pos3[j][2])*70;
    } else {
      const N=21; names=null; D=Array.from({length:N},()=>new Array(N));
      const pos3=[]; for(let k=0;k<3;k++) for(let i=0;i<7;i++){ pos3.push([k*3+randn()*.3, randn()*.3, randn()*.3]); }
      for(let i=0;i<N;i++) for(let j=0;j<N;j++) D[i][j]=Math.hypot(pos3[i][0]-pos3[j][0],pos3[i][1]-pos3[j][1],pos3[i][2]-pos3[j][2])*60;
    }
    init();
  }
  function init(){
    const w=c.clientWidth, h=c.clientHeight||420;
    pos=D.map(()=>[w/2+randn()*100, h/2+randn()*100]);
    draw();
  }
  function step(){
    const n=D.length;
    const newPos=pos.map(p=>p.slice());
    let stress=0;
    for(let i=0;i<n;i++){
      let nx=0, ny=0, ws=0;
      for(let j=0;j<n;j++){
        if(i===j) continue;
        const dx=pos[i][0]-pos[j][0], dy=pos[i][1]-pos[j][1];
        const dij=Math.hypot(dx,dy)+1e-9;
        const ratio = D[i][j]/dij;
        nx += pos[j][0] + ratio*(pos[i][0]-pos[j][0]);
        ny += pos[j][1] + ratio*(pos[i][1]-pos[j][1]);
        ws++;
        stress += (D[i][j]-dij)**2;
      }
      newPos[i]=[nx/ws, ny/ws];
    }
    pos=newPos;
    stressEl.textContent=stress.toFixed(0);
    draw();
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(!pos) return;
    // edges (faint)
    g.strokeStyle='rgba(124,196,255,.12)'; g.lineWidth=1;
    for(let i=0;i<pos.length;i++) for(let j=i+1;j<pos.length;j++){
      g.beginPath(); g.moveTo(pos[i][0],pos[i][1]); g.lineTo(pos[j][0],pos[j][1]); g.stroke();
    }
    pos.forEach((p,i)=>{
      drawPoint(g,p[0],p[1],i%5,6);
      if(names){ g.fillStyle='#fff'; g.font='11px monospace'; g.fillText(names[i], p[0]+8, p[1]+4); }
    });
  }
  dataSel.addEventListener('change',build);
  document.getElementById('mds-step').onclick=()=>{ if(!timer) timer=setInterval(step,80); };
  document.getElementById('mds-pause').onclick=()=>{clearInterval(timer); timer=null;};
  document.getElementById('mds-reset').onclick=init;
  MOUNTED.mds = () => { if(!D){ build(); } else draw(); };
})();

/* ====================================================
   k-Means
==================================================== */
(function(){
  const c=document.getElementById('km-c');
  const kEl=document.getElementById('km-k'),kv=document.getElementById('km-kv');
  const initSel=document.getElementById('km-init');
  const iterEl=document.getElementById('km-iter'),wssEl=document.getElementById('km-wss');
  let pts=[], cents=[], iter=0;
  function genData(){
    pts=[]; const w=c.clientWidth, h=c.clientHeight||420;
    const K=+kEl.value+1; // make a few clusters
    for(let k=0;k<K;k++){
      const cx=w*(.2+.6*rng()), cy=h*(.2+.6*rng());
      const n=20+Math.floor(rng()*15);
      for(let i=0;i<n;i++) pts.push({x:cx+randn()*25, y:cy+randn()*25});
    }
  }
  function initCents(){
    const k=+kEl.value; cents=[];
    if(initSel.value==='kpp' && pts.length){
      // Seeding via the tested k-means++ core in ./src/kmeans.js (shares the page rng).
      cents = kmeansPlusPlusInit(pts, k, rng);
    } else {
      const w=c.clientWidth, h=c.clientHeight||420;
      for(let i=0;i<k;i++) cents.push({x:w*(.2+.6*rng()), y:h*(.2+.6*rng())});
    }
    iter=0; iterEl.textContent=0; wssEl.textContent='—';
  }
  // Assignment step delegates to the tested k-means core in ./src/kmeans.js.
  function assign(){ return kmAssign(pts, cents); }
  function step(){
    if(!cents.length) initCents();
    const a=assign();
    wssEl.textContent=kmWss(pts, a, cents).toFixed(0);
    cents = kmUpdate(pts, a, cents.length, cents);
    iter++; iterEl.textContent=iter;
    draw();
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(cents.length){
      const a=assign();
      const step=10;
      for(let x=0;x<w;x+=step) for(let y=0;y<h;y+=step){
        let m=0,md=Infinity; cents.forEach((cc,i)=>{ const d=(cc.x-x)**2+(cc.y-y)**2; if(d<md){md=d;m=i;}});
        g.fillStyle=classColorAlpha(m,.10); g.fillRect(x,y,step,step);
      }
      pts.forEach((p,i)=>drawPoint(g,p.x,p.y,a[i],5));
      cents.forEach((cc,i)=>{
        g.strokeStyle='#fff'; g.lineWidth=2;
        g.beginPath(); g.moveTo(cc.x-8,cc.y); g.lineTo(cc.x+8,cc.y); g.moveTo(cc.x,cc.y-8); g.lineTo(cc.x,cc.y+8); g.stroke();
        g.fillStyle=classColor(i); g.beginPath(); g.arc(cc.x,cc.y,5,0,Math.PI*2); g.fill();
      });
    } else pts.forEach(p=>drawPoint(g,p.x,p.y,0,5));
  }
  bindClick(c,(x,y)=>{ pts.push({x,y}); draw(); });
  kEl.addEventListener('input',()=>{kv.textContent=kEl.value; initCents(); draw();});
  initSel.addEventListener('change',()=>{ initCents(); draw(); });
  document.getElementById('km-step').onclick=step;
  document.getElementById('km-run').onclick=async()=>{
    let prev=Infinity;
    for(let i=0;i<30;i++){
      step();
      const cur=+wssEl.textContent;
      if(Math.abs(prev-cur)<1e-2) break; prev=cur;
      await new Promise(r=>setTimeout(r,180));
    }
  };
  document.getElementById('km-demo').onclick=()=>{ genData(); initCents(); draw(); };
  document.getElementById('km-reset').onclick=()=>{ initCents(); draw(); };
  MOUNTED.kmeans = () => { if(!pts.length){ genData(); initCents(); } draw(); };
})();

/* ====================================================
   Logistic Regression
==================================================== */
(function(){
  const c=document.getElementById('lg-c'), cs=document.getElementById('lg-sig');
  const lr=document.getElementById('lg-lr'),lrv=document.getElementById('lg-lrv');
  const l2=document.getElementById('lg-l2'),l2v=document.getElementById('lg-l2v');
  const clsEl=document.getElementById('lg-cls');
  const lossEl=document.getElementById('lg-loss'),accEl=document.getElementById('lg-acc');
  let pts=[], w=[0,0,0], timer=null, hover={x:0,y:0};
  const W=()=>c.clientWidth, H=()=>c.clientHeight||380;
  function sig(z){ return 1/(1+Math.exp(-z)); }
  function fx(p){ const xn=(p.x/W())*2-1, yn=(p.y/H())*2-1; return w[0]*xn+w[1]*yn+w[2]; }
  function train(){
    if(!pts.length) return;
    // One gradient step via the tested logistic-regression core in ./src/logreg.js.
    // Points are mapped to normalised [-1,1] coordinates first.
    const norm = pts.map(p=>({ x:(p.x/W())*2-1, y:(p.y/H())*2-1, c:p.c }));
    const res = logregStep(w, norm, { lr:+lr.value, l2:+l2.value });
    w = res.weights;
    lossEl.textContent=res.loss.toFixed(3);
    accEl.textContent=(res.accuracy*100).toFixed(1)+'%';
    draw();
  }
  function draw(){
    const {g,w:cw,h:ch}=fitCanvas(c); gridBg(g,cw,ch);
    const step=8;
    for(let x=0;x<cw;x+=step) for(let y=0;y<ch;y+=step){
      const p=sig(fx({x,y}));
      g.fillStyle = p>0.5 ? `rgba(255,124,156,${.10+(p-.5)*.6})` : `rgba(124,196,255,${.10+(.5-p)*.6})`;
      g.fillRect(x,y,step,step);
    }
    // 0.5 line: w0*xn + w1*yn + w2 = 0 → yn = -(w0*xn+w2)/w1
    g.strokeStyle='#fff'; g.lineWidth=1.5; g.beginPath();
    if(Math.abs(w[1])>1e-6){
      for(let x=0;x<cw;x+=2){
        const xn=(x/cw)*2-1; const yn=-(w[0]*xn+w[2])/w[1];
        const y=(yn+1)/2*ch;
        if(x===0) g.moveTo(x,y); else g.lineTo(x,y);
      }
      g.stroke();
    }
    pts.forEach(p=>drawPoint(g,p.x,p.y,p.c,5));
    // sigmoid panel
    const {g:gs,w:sw,h:sh}=fitCanvas(cs); gridBg(gs,sw,sh);
    gs.strokeStyle='#2a3a63'; gs.beginPath(); gs.moveTo(0,sh/2); gs.lineTo(sw,sh/2); gs.moveTo(sw/2,0); gs.lineTo(sw/2,sh); gs.stroke();
    gs.strokeStyle='#7cc4ff'; gs.lineWidth=2; gs.beginPath();
    const sx=sw/12, sy=sh*.8;
    for(let px=0;px<sw;px++){ const z=(px-sw/2)/sx; const y=sh/2 - (sig(z)-0.5)*sy*2*0.4; if(px===0) gs.moveTo(px,y); else gs.lineTo(px,y); }
    gs.stroke();
    // marker at hovered point
    const z=fx(hover); const px=sw/2+z*sx; const py=sh/2-(sig(z)-0.5)*sy*0.8;
    gs.fillStyle='#ffb86b'; gs.beginPath(); gs.arc(px,py,5,0,Math.PI*2); gs.fill();
    gs.fillStyle='#8b96a8'; gs.font='11px monospace';
    gs.fillText(`z=${z.toFixed(2)}  σ(z)=${sig(z).toFixed(3)}`, 10, 16);
  }
  bindClick(c,(x,y)=>{ pts.push({x,y,c:+clsEl.value}); draw(); });
  c.addEventListener('mousemove',e=>{ const r=c.getBoundingClientRect(); hover={x:e.clientX-r.left,y:e.clientY-r.top}; draw(); });
  lr.addEventListener('input',()=>lrv.textContent=(+lr.value).toFixed(2));
  l2.addEventListener('input',()=>l2v.textContent=(+l2.value).toFixed(2));
  document.getElementById('lg-run').onclick=()=>{ if(!timer) timer=setInterval(train,40); };
  document.getElementById('lg-pause').onclick=()=>{clearInterval(timer);timer=null;};
  document.getElementById('lg-demo').onclick=()=>{
    pts=[]; const ww=W(),hh=H();
    for(let i=0;i<25;i++) pts.push({x:ww*.3+randn()*30, y:hh*.6+randn()*30, c:0});
    for(let i=0;i<25;i++) pts.push({x:ww*.7+randn()*30, y:hh*.4+randn()*30, c:1});
    w=[randn()*.1,randn()*.1,0]; draw();
  };
  document.getElementById('lg-clear').onclick=()=>{pts=[];w=[0,0,0];draw();};
  MOUNTED.logreg=()=>{ if(!pts.length) document.getElementById('lg-demo').onclick(); else draw(); };
})();

/* ====================================================
   Cross-Validation
==================================================== */
(function(){
  const c=document.getElementById('cv-c');
  const kEl=document.getElementById('cv-k'),kv=document.getElementById('cv-kv');
  const fEl=document.getElementById('cv-f'),fv=document.getElementById('cv-fv');
  const dEl=document.getElementById('cv-d'),dv=document.getElementById('cv-dv');
  const mseEl=document.getElementById('cv-mse'),cvEl=document.getElementById('cv-cv');
  let data=[];
  function trueF(x){ return Math.sin(x*Math.PI*1.2)*0.6 + 0.2*x; }
  function gen(){
    data=[]; for(let i=0;i<40;i++){ const x=(i/39)*2-1; data.push({x,y:trueF(x)+randn()*0.15}); }
    fEl.max=+kEl.value-1; fEl.value=0;
    draw();
  }
  function polyFit(pts,deg){
    const X=pts.map(p=>{ const r=[]; for(let j=0;j<=deg;j++) r.push(Math.pow(p.x,j)); return r;});
    const y=pts.map(p=>p.y); const XT=transpose(X), XTX=matmul(XT,X);
    for(let i=0;i<XTX.length;i++) XTX[i][i]+=1e-6;
    return solve(XTX, matvec(XT,y));
  }
  function evalPoly(w,x){ let y=0; for(let j=0;j<w.length;j++) y+=w[j]*Math.pow(x,j); return y; }
  function folds(){
    const k=+kEl.value;
    const idx=data.map((_,i)=>i);
    const sz=Math.ceil(idx.length/k);
    const out=[]; for(let i=0;i<k;i++) out.push(idx.slice(i*sz,(i+1)*sz));
    return out;
  }
  function draw(){
    const {g,w:cw,h:ch}=fitCanvas(c); gridBg(g,cw,ch);
    function px(x){ return (x+1)/2*cw; }
    function py(y){ return (1-(y+1.2)/2.4)*ch; }
    // true f
    g.strokeStyle='#ffb86b'; g.lineWidth=1.5; g.setLineDash([3,3]); g.beginPath();
    for(let i=0;i<=200;i++){ const x=-1+i/100, Y=trueF(x); if(i===0) g.moveTo(px(x),py(Y)); else g.lineTo(px(x),py(Y)); }
    g.stroke(); g.setLineDash([]);
    const F=folds(); const cf=Math.min(+fEl.value, F.length-1);
    const valSet = F[cf].map(i=>data[i]);
    const trSet  = data.filter((_,i)=>!F[cf].includes(i));
    if(trSet.length>0){
      const w=polyFit(trSet, +dEl.value);
      g.strokeStyle='#7cffb2'; g.lineWidth=2; g.beginPath();
      for(let i=0;i<=200;i++){ const x=-1+i/100, Y=evalPoly(w,x); if(i===0) g.moveTo(px(x),py(Y)); else g.lineTo(px(x),py(Y)); }
      g.stroke();
      let tr=0, vl=0;
      trSet.forEach(p=>{tr+=(p.y-evalPoly(w,p.x))**2;}); tr/=trSet.length;
      valSet.forEach(p=>{vl+=(p.y-evalPoly(w,p.x))**2;}); vl/=valSet.length;
      mseEl.textContent = vl.toFixed(3);
    }
    data.forEach((p,i)=>{
      const isVal=F[cf].includes(i);
      g.fillStyle = isVal ? '#ffb86b' : '#7cc4ff';
      g.beginPath(); g.arc(px(p.x),py(p.y), isVal?6:4, 0, Math.PI*2); g.fill();
      g.strokeStyle='rgba(0,0,0,.5)'; g.stroke();
    });
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`fold ${cf+1}/${F.length}  train=${data.length-F[cf].length}  val=${F[cf].length}`,10,16);
  }
  function runAll(){
    const F=folds(); let sum=0;
    F.forEach(idx=>{
      const tr=data.filter((_,i)=>!idx.includes(i));
      const vl=idx.map(i=>data[i]);
      const w=polyFit(tr,+dEl.value);
      let m=0; vl.forEach(p=>m+=(p.y-evalPoly(w,p.x))**2); sum+=m/vl.length;
    });
    cvEl.textContent=(sum/F.length).toFixed(3);
  }
  kEl.addEventListener('input',()=>{kv.textContent=kEl.value; fEl.max=+kEl.value-1; if(+fEl.value>+fEl.max) fEl.value=0; fv.textContent=fEl.value; draw();});
  fEl.addEventListener('input',()=>{fv.textContent=fEl.value; draw();});
  dEl.addEventListener('input',()=>{dv.textContent=dEl.value; draw();});
  document.getElementById('cv-run').onclick=runAll;
  document.getElementById('cv-demo').onclick=gen;
  MOUNTED.cv = () => { if(!data.length) gen(); else draw(); };
})();

/* ====================================================
   ROC / PR / Confusion
==================================================== */
(function(){
  const ch=document.getElementById('rc-hist'),cr=document.getElementById('rc-roc'),cp=document.getElementById('rc-pr'),cm=document.getElementById('rc-cm');
  const tEl=document.getElementById('rc-t'),tv=document.getElementById('rc-tv');
  const sEl=document.getElementById('rc-s'),sv=document.getElementById('rc-sv');
  let scores=[]; // {score, y}
  function gen(){
    scores=[]; const sep=+sEl.value;
    for(let i=0;i<120;i++) scores.push({score:1/(1+Math.exp(-(randn()-sep/2))), y:0});
    for(let i=0;i<120;i++) scores.push({score:1/(1+Math.exp(-(randn()+sep/2))), y:1});
    draw();
  }
  function metricsAt(t){
    let tp=0,fp=0,tn=0,fn=0;
    scores.forEach(s=>{ const p=s.score>=t?1:0; if(s.y===1&&p===1) tp++; else if(s.y===0&&p===1) fp++; else if(s.y===0&&p===0) tn++; else fn++; });
    return {tp,fp,tn,fn};
  }
  function curves(){
    const ts=[]; for(let i=0;i<=100;i++) ts.push(i/100);
    const roc=ts.map(t=>{const m=metricsAt(t); return {t,fpr:m.fp/(m.fp+m.tn+1e-9), tpr:m.tp/(m.tp+m.fn+1e-9)};});
    const pr =ts.map(t=>{const m=metricsAt(t); return {t,rec:m.tp/(m.tp+m.fn+1e-9), pre:(m.tp+m.fp)?m.tp/(m.tp+m.fp):1};});
    return {roc,pr};
  }
  function auc(roc){
    // sort by fpr asc
    const r=roc.slice().sort((a,b)=>a.fpr-b.fpr); let a=0;
    for(let i=1;i<r.length;i++) a += (r[i].fpr-r[i-1].fpr)*(r[i].tpr+r[i-1].tpr)/2;
    return a;
  }
  function draw(){
    if(!scores.length) gen();
    const t=+tEl.value;
    // histogram
    {
      const {g,w,h}=fitCanvas(ch); gridBg(g,w,h);
      const bins=20; const h0=new Array(bins).fill(0), h1=new Array(bins).fill(0);
      scores.forEach(s=>{ const b=Math.min(bins-1,Math.floor(s.score*bins)); if(s.y===0) h0[b]++; else h1[b]++; });
      const mx=Math.max(...h0,...h1,1);
      const bw=w/bins;
      for(let i=0;i<bins;i++){
        g.fillStyle='rgba(124,196,255,.6)'; g.fillRect(i*bw+1, h-20-h0[i]/mx*(h-30), bw-2, h0[i]/mx*(h-30));
        g.fillStyle='rgba(255,124,156,.6)'; g.fillRect(i*bw+1, h-20-h1[i]/mx*(h-30), bw-2, h1[i]/mx*(h-30));
      }
      // threshold line
      g.strokeStyle='#fff'; g.setLineDash([3,3]); g.beginPath(); g.moveTo(t*w,0); g.lineTo(t*w,h-20); g.stroke(); g.setLineDash([]);
      g.fillStyle='#8b96a8'; g.font='11px monospace'; g.fillText(`τ=${t.toFixed(2)}`, t*w+4, 14);
    }
    const C=curves(); const AUC=auc(C.roc);
    // ROC
    {
      const {g,w,h}=fitCanvas(cr); gridBg(g,w,h);
      g.strokeStyle='#2a3a63'; g.beginPath(); g.moveTo(0,h); g.lineTo(w,0); g.stroke();
      g.strokeStyle='#7cffb2'; g.lineWidth=2; g.beginPath();
      C.roc.slice().sort((a,b)=>a.fpr-b.fpr).forEach((p,i)=>{ const x=p.fpr*w, y=h-p.tpr*h; if(i===0) g.moveTo(x,y); else g.lineTo(x,y); });
      g.stroke();
      // current point
      const m=metricsAt(t); const x=m.fp/(m.fp+m.tn+1e-9)*w, y=h-m.tp/(m.tp+m.fn+1e-9)*h;
      g.fillStyle='#ffb86b'; g.beginPath(); g.arc(x,y,5,0,Math.PI*2); g.fill();
      g.fillStyle='#8b96a8'; g.font='11px monospace'; g.fillText(`FPR`, w-30, h-6); g.fillText('TPR', 6, 14);
      g.fillText(`AUC=${AUC.toFixed(3)}`, 6, h-8);
    }
    // PR
    {
      const {g,w,h}=fitCanvas(cp); gridBg(g,w,h);
      g.strokeStyle='#ff7c9c'; g.lineWidth=2; g.beginPath();
      C.pr.slice().sort((a,b)=>a.rec-b.rec).forEach((p,i)=>{ const x=p.rec*w, y=h-p.pre*h; if(i===0) g.moveTo(x,y); else g.lineTo(x,y); });
      g.stroke();
      const m=metricsAt(t); const rec=m.tp/(m.tp+m.fn+1e-9), pre=(m.tp+m.fp)?m.tp/(m.tp+m.fp):1;
      g.fillStyle='#ffb86b'; g.beginPath(); g.arc(rec*w,h-pre*h,5,0,Math.PI*2); g.fill();
      g.fillStyle='#8b96a8'; g.font='11px monospace'; g.fillText('Recall', w-40, h-6); g.fillText('Precision', 6, 14);
    }
    // CM
    {
      const {g,w,h}=fitCanvas(cm); g.fillStyle='#0f1623'; g.fillRect(0,0,w,h);
      const m=metricsAt(t);
      const cells=[[m.tn,'TN',m.tn,'#7cc4ff',1,1],[m.fp,'FP',m.fp,'#ff7c9c',2,1],[m.fn,'FN',m.fn,'#ff7c9c',1,2],[m.tp,'TP',m.tp,'#7cffb2',2,2]];
      const cw=w/3, chx=h/3;
      g.fillStyle='#8b96a8'; g.font='11px monospace'; g.textAlign='center';
      g.fillText('Pred 0', cw*1.5, chx*0.5); g.fillText('Pred 1', cw*2.5, chx*0.5);
      g.save(); g.translate(cw*0.5, chx*1.5); g.rotate(-Math.PI/2); g.fillText('True 0', 0,0); g.restore();
      g.save(); g.translate(cw*0.5, chx*2.5); g.rotate(-Math.PI/2); g.fillText('True 1', 0,0); g.restore();
      const total=m.tn+m.fp+m.fn+m.tp;
      cells.forEach(([v,label,_,col,col2,row])=>{
        const x=col2*cw+4, y=row*chx+4; const wd=cw-8, ht=chx-8;
        g.fillStyle = col + '33'; g.fillRect(x,y,wd,ht);
        g.strokeStyle=col; g.strokeRect(x,y,wd,ht);
        g.fillStyle='#e6edf3'; g.font='bold 18px monospace'; g.textAlign='center';
        g.fillText(v, x+wd/2, y+ht/2+4);
        g.font='10px monospace'; g.fillStyle='#8b96a8'; g.fillText(label, x+wd/2, y+14);
      });
      g.textAlign='start';
      const acc=(m.tp+m.tn)/Math.max(1,total), pre=(m.tp+m.fp)?m.tp/(m.tp+m.fp):0;
      const rec=m.tp/(m.tp+m.fn+1e-9), f1=pre+rec?2*pre*rec/(pre+rec):0;
      document.getElementById('rc-acc').textContent=(acc*100).toFixed(1)+'%';
      document.getElementById('rc-pre').textContent=pre.toFixed(3);
      document.getElementById('rc-rec').textContent=rec.toFixed(3);
      document.getElementById('rc-f1').textContent=f1.toFixed(3);
      document.getElementById('rc-auc').textContent=AUC.toFixed(3);
    }
  }
  tEl.addEventListener('input',()=>{tv.textContent=(+tEl.value).toFixed(2); draw();});
  sEl.addEventListener('input',()=>{sv.textContent=(+sEl.value).toFixed(1); gen();});
  document.getElementById('rc-demo').onclick=gen;
  MOUNTED.roc = () => { if(!scores.length) gen(); else draw(); };
})();

/* ====================================================
   Hierarchical Clustering
==================================================== */
(function(){
  const c=document.getElementById('hc-c'), cd=document.getElementById('hc-d');
  const link=document.getElementById('hc-link');
  const hEl=document.getElementById('hc-h'),hv=document.getElementById('hc-hv');
  const ncEl=document.getElementById('hc-nc');
  let pts=[]; let merges=[]; // {a,b,dist}
  function dist(A,B){ return Math.hypot(A.x-B.x, A.y-B.y); }
  function clusterDist(A,B){
    const m=link.value;
    if(m==='single'){ let d=Infinity; for(const a of A.pts) for(const b of B.pts) d=Math.min(d,dist(a,b)); return d; }
    if(m==='complete'){ let d=0; for(const a of A.pts) for(const b of B.pts) d=Math.max(d,dist(a,b)); return d; }
    if(m==='ward'){ // approx: distance between centroids weighted
      const dx=A.cx-B.cx, dy=A.cy-B.cy; const nA=A.pts.length, nB=B.pts.length;
      return Math.sqrt(nA*nB/(nA+nB))*Math.hypot(dx,dy);
    }
    // average
    let d=0,n=0; for(const a of A.pts) for(const b of B.pts){ d+=dist(a,b); n++; } return n?d/n:0;
  }
  function cluster(){
    merges=[];
    let cs=pts.map((p,i)=>({pts:[p], cx:p.x, cy:p.y, id:i}));
    while(cs.length>1){
      let bi=0,bj=1,bd=Infinity;
      for(let i=0;i<cs.length;i++) for(let j=i+1;j<cs.length;j++){
        const d=clusterDist(cs[i],cs[j]); if(d<bd){bd=d;bi=i;bj=j;}
      }
      const A=cs[bi], B=cs[bj];
      const merged={pts:A.pts.concat(B.pts),
        cx:(A.cx*A.pts.length+B.cx*B.pts.length)/(A.pts.length+B.pts.length),
        cy:(A.cy*A.pts.length+B.cy*B.pts.length)/(A.pts.length+B.pts.length),
        id: cs.length+merges.length+1};
      merges.push({a:A,b:B,dist:bd,merged});
      cs=cs.filter((_,k)=>k!==bi&&k!==bj); cs.push(merged);
    }
  }
  function flatClusters(cutH){
    // walk merges in reverse, undoing ones above cutH
    if(!pts.length) return [];
    let groups = [pts.map((_,i)=>i)]; // start fully merged: all one group
    // Build by going through merges chronologically and using a Union-Find that only unions if dist<=cutH
    const parent=pts.map((_,i)=>i);
    function find(i){ while(parent[i]!==i){parent[i]=parent[parent[i]]; i=parent[i];} return i; }
    function union(i,j){ const ri=find(i),rj=find(j); if(ri!==rj) parent[ri]=rj; }
    for(const m of merges){
      if(m.dist>cutH) break;
      // union representative points of A and B
      const ia=pts.indexOf(m.a.pts[0]), ib=pts.indexOf(m.b.pts[0]);
      union(ia,ib);
    }
    const map=new Map(); pts.forEach((_,i)=>{ const r=find(i); if(!map.has(r)) map.set(r,map.size); });
    return pts.map((_,i)=>map.get(find(i)));
  }
  function draw(){
    if(pts.length>1) cluster();
    const cut=+hEl.value;
    const labels = pts.length ? flatClusters(cut) : [];
    ncEl.textContent = labels.length? new Set(labels).size : '—';
    // canvas: points + cluster edges below cut
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    g.strokeStyle='rgba(255,255,255,.18)'; g.lineWidth=1;
    for(const m of merges){
      if(m.dist>cut) break;
      g.beginPath(); g.moveTo(m.a.cx,m.a.cy); g.lineTo(m.b.cx,m.b.cy); g.stroke();
    }
    pts.forEach((p,i)=>drawPoint(g,p.x,p.y,labels[i]||0,5));
    // dendrogram
    const {g:gd,w:dw,h:dh}=fitCanvas(cd); gridBg(gd,dw,dh);
    if(!pts.length){ gd.fillStyle='#8b96a8'; gd.font='12px monospace'; gd.fillText('Add points to see dendrogram',10,20); return; }
    const N=pts.length, leafX=new Map();
    pts.forEach((p,i)=> leafX.set(i, 20 + (i/(N-1+0.001))*(dw-40)));
    const maxD = merges.length? merges[merges.length-1].dist : 1;
    function yFor(d){ return dh-20 - d/maxD*(dh-50); }
    gd.strokeStyle='#2a3a63'; gd.beginPath(); gd.moveTo(20,dh-20); gd.lineTo(dw-20,dh-20); gd.stroke();
    // For each merge, draw bracket; compute x position as midpoint of representative leaves
    const repX=new Map(); pts.forEach((_,i)=>repX.set('p'+i, leafX.get(i)));
    // Track group representative x
    const groupX=new Map(); merges.forEach((m,k)=>{
      const ka='p'+pts.indexOf(m.a.pts[0]);
      const kb='p'+pts.indexOf(m.b.pts[0]);
      const xa = repX.get(ka) ?? groupX.get(ka), xb=repX.get(kb) ?? groupX.get(kb);
      const y=yFor(m.dist), ya=yFor(0), yb=yFor(0); // simplistic
      gd.strokeStyle = m.dist<=cut ? '#7cffb2' : '#7cc4ff'; gd.lineWidth=1.5;
      gd.beginPath();
      gd.moveTo(xa, dh-20); gd.lineTo(xa, y); gd.lineTo(xb, y); gd.lineTo(xb, dh-20);
      gd.stroke();
      const newKey = 'm'+k;
      groupX.set('p'+pts.indexOf(m.a.pts[0]), (xa+xb)/2);
      groupX.set('p'+pts.indexOf(m.b.pts[0]), (xa+xb)/2);
    });
    // cut line
    gd.strokeStyle='#ffb86b'; gd.setLineDash([4,3]); gd.beginPath();
    gd.moveTo(20,yFor(cut)); gd.lineTo(dw-20,yFor(cut)); gd.stroke(); gd.setLineDash([]);
    gd.fillStyle='#ffb86b'; gd.font='11px monospace'; gd.fillText(`cut=${cut}`, 24, yFor(cut)-4);
  }
  bindClick(c,(x,y)=>{ pts.push({x,y}); draw(); });
  link.addEventListener('change',draw);
  hEl.addEventListener('input',()=>{hv.textContent=hEl.value; draw();});
  document.getElementById('hc-demo').onclick=()=>{
    pts=[]; const w=c.clientWidth, h=c.clientHeight||380;
    [[.25,.3],[.75,.3],[.5,.75]].forEach(([fx,fy])=>{
      for(let i=0;i<10;i++) pts.push({x:fx*w+randn()*22, y:fy*h+randn()*22});
    });
    draw();
  };
  document.getElementById('hc-clear').onclick=()=>{pts=[];merges=[];draw();};
  MOUNTED.hc = () => { if(!pts.length) document.getElementById('hc-demo').onclick(); else draw(); };
})();

/* ====================================================
   Anomaly Detection
==================================================== */
(function(){
  const c=document.getElementById('anom-c');
  const mEl=document.getElementById('anom-m');
  const ccEl=document.getElementById('anom-cc'),ccv=document.getElementById('anom-cv');
  const kEl=document.getElementById('anom-k'),kv=document.getElementById('anom-kv');
  const outEl=document.getElementById('anom-out');
  let pts=[];
  function stats(){
    const n=pts.length; if(!n) return null;
    let mx=0,my=0; pts.forEach(p=>{mx+=p.x;my+=p.y;}); mx/=n; my/=n;
    let sxx=0,syy=0,sxy=0;
    pts.forEach(p=>{const dx=p.x-mx, dy=p.y-my; sxx+=dx*dx; syy+=dy*dy; sxy+=dx*dy;});
    sxx/=n; syy/=n; sxy/=n;
    const det=sxx*syy-sxy*sxy + 1e-6;
    const inv=[[syy/det,-sxy/det],[-sxy/det,sxx/det]];
    return {mx,my,sxx,syy,sxy,det,inv};
  }
  function score(p, s){
    if(!s) return 0;
    const m=mEl.value;
    if(m==='knn'){
      const k=+kEl.value;
      const ds=pts.map(q=>Math.hypot(p.x-q.x,p.y-q.y)).sort((a,b)=>a-b).slice(0,k);
      return ds.reduce((a,b)=>a+b,0)/k;
    }
    const dx=p.x-s.mx, dy=p.y-s.my;
    if(m==='maha'){
      return Math.sqrt(Math.max(0, dx*(s.inv[0][0]*dx+s.inv[0][1]*dy) + dy*(s.inv[1][0]*dx+s.inv[1][1]*dy)));
    }
    // gauss: -log likelihood
    const q = dx*(s.inv[0][0]*dx+s.inv[0][1]*dy) + dy*(s.inv[1][0]*dx+s.inv[1][1]*dy);
    return 0.5*q + 0.5*Math.log(s.det);
  }
  function draw(){
    const {g,w,h}=fitCanvas(c); gridBg(g,w,h);
    if(!pts.length) return;
    const s=stats();
    // build heatmap & per-point scores
    const sc = pts.map(p=>score(p,s));
    const sorted=sc.slice().sort((a,b)=>b-a);
    const idx=Math.floor((+ccEl.value)/100*sorted.length);
    const cutoff = sorted[Math.min(idx, sorted.length-1)] ?? Infinity;
    let mn=Infinity, mx=-Infinity;
    const grid=20, vals=[];
    for(let i=0;i<=grid;i++){ vals[i]=[];
      for(let j=0;j<=grid;j++){
        const v=score({x:i/grid*w, y:j/grid*h}, s);
        vals[i][j]=v; if(v<mn) mn=v; if(v>mx) mx=v;
      }
    }
    const cellW=w/grid, cellH=h/grid;
    for(let i=0;i<grid;i++) for(let j=0;j<grid;j++){
      const v=vals[i][j], t=(v-mn)/(mx-mn+1e-9);
      g.fillStyle = `rgba(${Math.floor(40+t*215)},${Math.floor(60-t*40)},${Math.floor(120-t*60)},.55)`;
      g.fillRect(i*cellW, j*cellH, cellW+1, cellH+1);
    }
    let outCount=0;
    pts.forEach((p,i)=>{
      const isOut = sc[i] >= cutoff;
      if(isOut) outCount++;
      drawPoint(g,p.x,p.y, isOut?1:0, isOut?6:4);
      if(isOut){ g.strokeStyle='#ffb86b'; g.lineWidth=2; g.beginPath(); g.arc(p.x,p.y,11,0,Math.PI*2); g.stroke(); }
    });
    outEl.textContent = `${outCount}/${pts.length}`;
    g.fillStyle='#8b96a8'; g.font='12px monospace';
    g.fillText(`method=${mEl.value}  contamination=${ccEl.value}%`, 10, 16);
  }
  bindClick(c,(x,y)=>{ pts.push({x,y}); draw(); });
  [mEl,kEl,ccEl].forEach(el=>el.addEventListener('input',()=>{
    ccv.textContent=ccEl.value+'%'; kv.textContent=kEl.value; draw();
  }));
  document.getElementById('anom-demo').onclick=()=>{
    pts=[]; const w=c.clientWidth, h=c.clientHeight||420;
    for(let i=0;i<70;i++) pts.push({x:w/2+randn()*60, y:h/2+randn()*40});
    // a few outliers
    for(let i=0;i<6;i++) pts.push({x:w*(.1+rng()*.8), y:h*(.1+rng()*.8)});
    draw();
  };
  document.getElementById('anom-clear').onclick=()=>{pts=[];draw();};
  MOUNTED.anom = () => { if(!pts.length) document.getElementById('anom-demo').onclick(); else draw(); };
})();

/* ============ initial paint of intro ============ */
window.addEventListener('resize',()=>{
  const active=document.querySelector('.section.active');
  if(active && MOUNTED[active.id]) MOUNTED[active.id]();
});
})();
