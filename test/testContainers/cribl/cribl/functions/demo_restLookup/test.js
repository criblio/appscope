const http = require('http');
const agentOpts = { keepAlive: true, maxSockets: 10 };
const agent = http.Agent(agentOpts);

function lookup() {
    const u = 'http://localhost:3000/users/mwilde';
    return new Promise((resolve, reject) => {
        http.get(u, { agent }, (resp) => {
          let data = '';
    
          resp.on('data', (chunk) => {
            data += chunk;
          });
    
          resp.on('end', () => {
            let d = data;
            try {
              d = JSON.parse(data);
            } catch (e) {
            }
            // event[conf.eventField] = d;
            resolve();
          });
    
        }).on("error", (err) => {
        //   dLogger.error(`Error in REST Lookup: ${err.message}`);
          reject(`Error: ${err.message}`);
        });
    });
}


for (let i = 0; i < 10000; i++) {
    lookup().then(() => {});
}

