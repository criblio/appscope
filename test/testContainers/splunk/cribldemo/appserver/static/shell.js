const src = `${window.location.protocol}//${window.location.hostname}:27425/`;
// console.log('shell src: ', src);
document.getElementById('shell_iframe').src = src;
setTimeout(() => {
  document.getElementById('shell_iframe').contentWindow.postMessage({ command: 'tail -f /tmp/events.ndjson\n' }, src);
}, 2000);
