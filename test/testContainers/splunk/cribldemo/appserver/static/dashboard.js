const name = undefined;
const email = undefined;

/* pipelines */
Array.from(document.getElementsByClassName('pipeline-name')).forEach(element => {
  const pipelineId = element.id;
  element.href = `http://${window.location.hostname}:9000/pipelines/${pipelineId}`;
});

/* eslint-disable */

(function () {
  var w = window;
  var ic = w.Intercom;
  if (typeof ic === "function") {
    ic('reattach_activator');
    ic('update', intercomSettings);
  } else {
    var d = document;
    var i = function () {
      i.c(arguments)
    };
    i.q = [];
    i.c = function (args) {
      i.q.push(args)
    };
    w.Intercom = i;
    var loaded = false;
    function l() {
      if(loaded) return;
      loaded = true;
      var s = d.createElement('script');
      s.type = 'text/javascript';
      s.async = true; s.src = 'https://widget.intercom.io/widget/s8fj4krt';
      var x = d.getElementsByTagName('script')[0];
      x.parentNode.insertBefore(s, x);
    }
    if(d.readyState == 'complete') { // 'load' has already fired
      l();
    } else if (w.attachEvent) {
      w.attachEvent('onload', l);
    } else {
      w.addEventListener('load', l, false);
    }
    // last ditch effort to call load
    setTimeout(l, 5000);
  }
}
)()

const intercomSettings = {
  app_id: "s8fj4krt",
  name,
  email,
};
console.log('intercomSettings: ', intercomSettings);
window.Intercom('boot', intercomSettings);
window.Intercom('update');
