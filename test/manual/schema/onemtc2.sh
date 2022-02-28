SOURCEFILE="/tmp/http.out.9"
OUTFILE="/tmp/http.filtered.9"


LIST=""
LIST+="http.duration.client "
LIST+="http.req "
LIST+="http.resp.content_length "
LIST+="http.duration.server "
LIST+="dns.req "
LIST+="net.open "

rm -f $OUTFILE

for METRIC in $LIST; do
   grep $METRIC $SOURCEFILE | head -n1 >> $OUTFILE
done

