import {test,expect} from '@playwright/test';
test.beforeEach(async({request})=>{await request.post('http://127.0.0.1:18088/__reset');});
test('studio edits make a new revision through the actual proxy',async({page,request})=>{
  await page.goto('/studio');await page.getByRole('button',{name:'sample.mp4',exact:true}).click();
  await expect(page.getByLabel('Translation segment 1')).toHaveValue('Hola');
  await page.getByLabel('Translation segment 1').fill('Buenos días');
  await page.getByRole('button',{name:'Save translation and regenerate changed work'}).click();
  await expect(page.getByRole('heading',{name:'sample.mp4 · queued'})).toBeVisible();
  const recorded=await (await request.get('http://127.0.0.1:18088/__requests')).json();
  expect(recorded.revisions[0].translation).toEqual(['Buenos días']);
});
test('glossary validation and speaker voice selection',async({page,request})=>{
  await page.goto('/studio');await page.getByRole('button',{name:'sample.mp4',exact:true}).click();
  await page.getByLabel('Glossary').fill('{invalid');
  await page.getByRole('button',{name:'Apply options in a new version'}).click();
  await expect(page.getByRole('alert').filter({hasText:'Glossary must'})).toContainText('Glossary must');
  await page.getByLabel('Glossary').fill('{"Xeon":"Xeon"}');
  await page.getByRole('button',{name:'Load available voices'}).click();
  await page.getByLabel('Voice for speaker_1',{exact:true}).selectOption('Voice B');
  await page.getByRole('button',{name:'Apply options in a new version'}).click();
  await expect(page.getByRole('heading',{name:'sample.mp4 · queued'})).toBeVisible();
  const recorded=await (await request.get('http://127.0.0.1:18088/__requests')).json();
  expect(recorded.revisions[0].options.speaker_voices.speaker_1).toBe('Voice B');
});
test('account cookie is HttpOnly and signout removes it',async({page,context})=>{
 await page.goto('/studio');await page.getByText('Account access',{exact:true}).click();
 await page.getByLabel('Access token').fill('test-token');await page.getByRole('button',{name:'Sign in',exact:true}).click();
 await expect.poll(async()=> (await context.cookies()).find(c=>c.name==='session-token')?.httpOnly).toBe(true);
 expect(await page.evaluate(()=>document.cookie)).not.toContain('test-token');
 await page.getByRole('button',{name:'Sign out'}).click();
 await expect.poll(async()=> (await context.cookies()).some(c=>c.name==='session-token')).toBe(false);
});
test('webcam permission failure returns to retryable idle state',async({page})=>{
 await page.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Camera denied','NotAllowedError');};});
 await page.goto('/live');await page.getByRole('button',{name:'Start recording'}).click();
 await expect(page.getByText('Camera denied',{exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'Start recording'})).toBeEnabled();
 await expect(page.getByRole('button',{name:'Stop & translate'})).toBeDisabled();
});
test('submission failure retains session for idempotent retry',async({page,request})=>{
 await page.addInitScript(()=>{
   navigator.mediaDevices.getUserMedia=async()=>new MediaStream();
   (window as any).RTCPeerConnection=class extends EventTarget{
     iceGatheringState='complete';connectionState='connected';localDescription={sdp:'fixture',type:'offer'};
     addTrack(){} async createOffer(){return this.localDescription;} async setLocalDescription(){} async setRemoteDescription(){}
     async getStats(){return new Map([['video',{type:'outbound-rtp',kind:'video',packetsSent:5}]]);}close(){}
   };
 });
 await page.goto('/live');await page.getByRole('button',{name:'Start recording'}).click();
 await expect(page.getByRole('button',{name:'Stop & translate'})).toBeEnabled();
 await page.getByRole('button',{name:'Stop & translate'}).click();
 await expect(page.getByText(/Temporary upstream failure/)).toBeVisible();
 await page.getByRole('button',{name:'Stop & translate'}).click();
 await expect(page.getByRole('button',{name:'Start recording'})).toBeEnabled();
 expect((await (await request.get('http://127.0.0.1:18088/__requests')).json()).stopCalls).toBe(2);
});
test('avatar requires portrait and exposes voices',async({page})=>{
 await page.goto('/avatar');await expect(page.getByRole('button',{name:'Start conversation'})).toBeDisabled();
 await page.getByRole('button',{name:'Load voices'}).click();await page.getByLabel('Avatar voice').selectOption('Voice A');
 await expect(page.getByLabel('Avatar voice')).toHaveValue('Voice A');
 await expect(page.getByRole('heading')).toHaveCSS('font-size','24px');
});
