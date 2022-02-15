SOURCEFILE="/tmp/http.out.9"
OUTFILE="/tmp/http.filtered.9"


LIST=""
LIST+="http.client.duration "
LIST+="http.requests "
LIST+="http.response.content_length "
LIST+="http.server.duration "
LIST+="net.dns "
LIST+="net.open "

rm -f $OUTFILE

for METRIC in $LIST; do
   grep $METRIC $SOURCEFILE | head -n1 >> $OUTFILE
done

