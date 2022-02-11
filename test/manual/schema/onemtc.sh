SOURCEFILE="/tmp/mtc.out"
OUTFILE="/tmp/mtc.filtered"

rm -f $OUTFILE
# Low verbosity
grep net.error $SOURCEFILE | grep connection | head -n1 >> $OUTFILE
grep net.error $SOURCEFILE | grep rx_tx      | head -n1 >> $OUTFILE
grep net.error $SOURCEFILE | grep dns        | head -n1 >> $OUTFILE
grep fs.error  $SOURCEFILE | grep open_close | head -n1 >> $OUTFILE
grep fs.error  $SOURCEFILE | grep read_write | head -n1 >> $OUTFILE
grep fs.error  $SOURCEFILE | grep stat       | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep inet_tcp   | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep inet_udp   | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep unix_tcp   | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep unix_udp   | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep other      | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep inet_tcp   | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep inet_udp   | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep unix_tcp   | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep unix_udp   | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep other      | head -n1 >> $OUTFILE

# High verbosity
grep net.rx    $SOURCEFILE | grep AF_INET | grep TCP | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep AF_INET | grep UDP | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep UNIX    | grep TCP | head -n1 >> $OUTFILE
grep net.rx    $SOURCEFILE | grep UNIX    | grep UDP | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep AF_INET | grep TCP | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep AF_INET | grep UDP | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep UNIX    | grep TCP | head -n1 >> $OUTFILE
grep net.tx    $SOURCEFILE | grep UNIX    | grep UDP | head -n1 >> $OUTFILE

LIST=""
LIST+="net.dns "
LIST+="dns.duration "
LIST+="fs.stat "
LIST+="fs.duration "
LIST+="fs.write "
LIST+="fs.read "
LIST+="fs.open "
LIST+="fs.close "
LIST+="fs.seek "
LIST+="net.port "
LIST+="net.tcp "
LIST+="net.udp "
LIST+="net.other "
LIST+="net.open "
LIST+="net.close "
LIST+="net.duration "
LIST+="http.requests "
LIST+="http.server.duration "
LIST+="http.client.duration "
LIST+="http.request.content_length "
LIST+="http.response.content_length "
LIST+="proc.cpu "
LIST+="proc.cpu_perc "
LIST+="proc.mem "
LIST+="proc.thread "
LIST+="proc.fd "
LIST+="proc.child "
LIST+="proc.start "

for METRIC in $LIST; do
   grep $METRIC $SOURCEFILE | head -n1 >> $OUTFILE
done

grep "Binary data detected" $SOURCEFILE | head -n1 >> $OUTFILE
grep "Truncated metrics"    $SOURCEFILE | head -n1 >> $OUTFILE
