import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests',fullyParallel:false,workers:1,
  use:{baseURL:'http://127.0.0.1:3188',headless:true},
  webServer:[
    {command:'node tests/api-fixture.cjs',url:'http://127.0.0.1:18088/auth/whoami',reuseExistingServer:false},
    {command:'npm run start -- --hostname 127.0.0.1 --port 3188',url:'http://127.0.0.1:3188',reuseExistingServer:false,env:{API_PROXY_TARGET:'http://127.0.0.1:18088',INGEST_PROXY_TARGET:'http://127.0.0.1:18088',INTERNAL_API_KEY:'test-internal-key'}},
  ],
});
