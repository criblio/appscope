#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "curl..."
echo ""
curl -s -k https://localhost/
echo ""
echo "wget..."
echo ""
wget -q -O - --no-check-certificate https://localhost/
echo ""
echo "Python..."
echo ""
python3 ${DIR}/https.py
echo ""
echo "Perl..."
echo ""
perl ${DIR}/https.pl
echo ""
echo "Go..."
echo ""
${DIR}/goHttps
echo ""
echo "Java..."
echo ""
cd ${DIR} && java HTTPS
