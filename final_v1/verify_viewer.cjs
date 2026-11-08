// Exercise the generated offline replay in Chromium.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require(process.env.CAPSTONE_PLAYWRIGHT_MODULE || "playwright");

async function main() {
  const input = path.resolve(process.argv[2]);
  const mode = process.argv[3];
  const browser = await chromium.launch({headless:true,executablePath:process.env.CAPSTONE_BROWSER_EXE});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1050}});
    const errors = [];
    page.on("pageerror",error=>errors.push(error.message));
    await page.goto(pathToFileURL(input).href);
    await page.locator("#case option").first().waitFor({state:"attached"});
    const info = await page.evaluate(()=>{
      const bundle=JSON.parse(document.querySelector("#flight-data").textContent);
      return {
        cases:bundle.cases.length,
        compare:bundle.cases.filter(c=>c.phase==="비교").length,
        stress:bundle.cases.filter(c=>c.phase==="명령 손실").length,
        interventionCase:bundle.cases.findIndex(c=>c.runs.some(r=>r.interventions.length)),
        stressCase:bundle.cases.findIndex(c=>c.phase==="명령 손실"),
        maps:Object.keys(bundle.maps).length,
        valid:bundle.cases.every(c=>bundle.maps[c.region]?.shapes.length && c.runs.every(r=>
          r.positions.length===r.duration/3+1 && atob(r.edges).length===r.edges_stride*r.duration*2 &&
          r.positions.every(row=>row.length===r.n*2)))
      };
    });
    assert(info.valid,"Invalid trajectory or edge data");
    if (mode!=="sample") assert.deepEqual([info.compare,info.stress,info.maps],[42,14,14]);
    assert(info.interventionCase>=0);
    await page.selectOption("#case",String(info.interventionCase));
    assert.match(await page.locator("#clock").innerText(),/^0\.0 /);
    await page.click("#next");
    assert.notEqual((await page.locator("#clock").innerText()).split(" / ")[0],"0.0");
    await page.click("#play");
    await page.waitForTimeout(400);
    await page.click("#play");
    await page.click("#start");
    assert.match(await page.locator("#clock").innerText(),/^0\.0 /);
    const incident = await page.evaluate(()=>{
      const bundle=JSON.parse(document.querySelector("#flight-data").textContent);
      const c=bundle.cases[Number(document.querySelector("#case").value)];
      const arm=c.runs.findIndex(r=>r.interventions.length);
      return {arm,time:c.runs[arm].interventions[0].time_s};
    });
    await page.selectOption("#right select",String(incident.arm));
    await page.evaluate(t=>{
      const slider=document.querySelector("#seek");
      slider.value=t; slider.dispatchEvent(new Event("input"));
    },incident.time);
    assert.match(await page.locator("#right .info").innerText(),/선제 이동|복구/);
    assert.equal(await page.locator("#right .spark").count(),1);
    const shots=path.join(path.dirname(input),"artifacts","viewer_qa");
    fs.mkdirSync(shots,{recursive:true});
    await page.screenshot({path:path.join(shots,`${mode}_intervention.png`),fullPage:true});
    if (info.stressCase>=0) {
      await page.selectOption("#case",String(info.stressCase));
      assert.match(await page.locator("#right select").locator("option:checked").innerText(),/자율 보정/);
      await page.evaluate(()=>{
        const payload=JSON.parse(document.querySelector("#flight-data").textContent);
        const c=payload.cases[Number(document.querySelector("#case").value)];
        const tick=c.runs[1].lost.findIndex(row=>row.length);
        const slider=document.querySelector("#seek");
        slider.value=tick*3; slider.dispatchEvent(new Event("input"));
      });
      assert.match(await page.locator("#right .info").innerText(),/미수신/);
    }
    await page.screenshot({path:path.join(shots,`${mode}_desktop.png`),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(shots,`${mode}_mobile.png`),fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({browser:browser.version(),...info,errors}));
  } finally {
    await browser.close();
  }
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
