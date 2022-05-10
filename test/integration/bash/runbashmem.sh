#!/mybin/mem/bash
HEADERS=$(free | sed -n 1p | gawk '{OFS = ","; print $1, $2, $3, $4, $5, $6}')
OUTFILE=/tmp/free.out

i=0
echo $HEADERS > $OUTFILE
while [[ $i -lt 100 ]]; do
    if [[ $i -ne 0 ]]; then
        sleep 1
    fi
    echo "Second $i"
    DATA=$(free | sed -n 2p | gawk '{OFS = ","; print $2, $3, $4, $5, $6, $7}')
    echo $DATA >> $OUTFILE
    ((i=i+1))
done

