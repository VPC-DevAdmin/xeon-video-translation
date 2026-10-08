// Deterministic HTTP fixture for browser/proxy contracts. No model output claims.
const http=require('node:http');
let revisions=[],stopCalls=0;
const original={job_id:'a'.repeat(32),status:'failed',input_filename:'sample.mp4',target_language:'es',options:{},created_at:'2026-10-03T00:00:00Z',stages:[],error:'Speech exceeded its slot.'};
const revised={...original,job_id:'b'.repeat(32),status:'queued',parent_job_id:original.job_id,error:null};
http.createServer(async(req,res)=>{
 let raw='';for await(const chunk of req)raw+=chunk;
 const path=req.url.split('?')[0];
 const send=(value,status=200)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(value));};
 if(path==='/__reset'){revisions=[];stopCalls=0;return send({});}
 if(path==='/__requests')return send({revisions,stopCalls});
 if(path==='/auth/whoami')return send({owner:'test'});
 if(path==='/jobs')return send({jobs:[original,...(revisions.length?[revised]:[])]});
 if(path==='/speech/voices')return send({voices:['Voice A','Voice B'],languages:['es','en']});
 if(path.endsWith('/revise')){revisions.push(JSON.parse(raw));return send({job_id:revised.job_id},201);}
 if(path.endsWith('/cancel'))return send({status:'cancelling'});
 if(path==='/config')return send({iceServers:[],maxSeconds:120});
 if(path.endsWith('/offer'))return send({sdp:'fixture-sdp',type:'answer'});
 if(path.endsWith('/stop'))return ++stopCalls===1?send({detail:'Temporary upstream failure'},503):send({job_id:original.job_id});
 if(path.startsWith('/sessions/'))return send({seconds:3,caption:'Provisional words',status:'recording'});
 if(path.endsWith('/stream')){res.writeHead(200,{'Content-Type':'text/event-stream'});return res.end('event: stream_end\ndata: {}\n\n');}
 if(path.endsWith('/artifacts'))return send({artifacts:[{name:'input.mp4'}]});
 if(path.endsWith('translation.json')||path.endsWith('transcript.json'))return send({segments:[{start:0,end:1,text:'Hola',speaker:'speaker_1'}]});
 if(path.endsWith('/input.mp4')){res.writeHead(200,{'Content-Type':'video/mp4'});return res.end();}
 if(req.method==='DELETE')return send({status:'deleted'});
 if(path.startsWith('/jobs/'))return send(path.includes(revised.job_id)?revised:original);
 send({detail:'fixture route missing'},404);
}).listen(18088,'127.0.0.1');
