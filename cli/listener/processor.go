package listener

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode"

	"github.com/criblio/scope/libscope"
)

const DNS_AVERAGE_LENGTH = 40
const SHANNON_THRESHOLD = 4.2

type domain string

// Return the suspicious packets
func (d domain) verfiyLength() int {
	splittedDomain := strings.Split(string(d), ".")
	packets := 0
	for _, dom := range splittedDomain {
		if len(dom) > DNS_AVERAGE_LENGTH {
			packets += 1
		}
	}
	return packets
}

func (d domain) verifyCapitalChar() int {
	capitalLet := 0
	for _, char := range d {
		if unicode.IsUpper(char) {
			capitalLet += 1
		}
	}
	return capitalLet
}

func (d domain) Entropy() float64 {
	frq := make(map[rune]float64)

	//get frequency of characters
	for _, i := range d {
		frq[i]++
	}
	var entropy float64
	for _, v := range frq {
		if v == 0 {
			continue
		}
		p := float64(v) / float64(len(d))
		entropy -= p * math.Log2(p)
	}

	return entropy
}

func (d domain) verifyHexChar() int {
	hexChar := 0
	for _, c := range d {
		if c > unicode.MaxASCII {
			hexChar += 1
		}
	}
	return hexChar
}

// Process incoming scope data.
func processScopeData(processChan <-chan libscope.EventBody, notifyChan chan<- string) {
	for {
		select {
		case processEvent := <-processChan:
			var dnsDomainData libscope.DnsDomainData
			sourceData, err := json.Marshal(processEvent.Data)
			if err != nil {
				fmt.Println("Error marshaling source structure:", err)
				continue
			}
			err = json.Unmarshal(sourceData, &dnsDomainData)
			if err != nil {
				fmt.Println("Error unmarshalling source structure:", err)
				continue
			}
			fullDomain := domain(dnsDomainData.Domain)
			ent := fullDomain.Entropy()
			lenpack := fullDomain.verfiyLength()
			capit := fullDomain.verifyCapitalChar()
			hexchar := fullDomain.verifyHexChar()
			var msg string = ""
			if lenpack > 0 {
				msg += fmt.Sprintf("- Long message found in %d packets\n", lenpack)
			}
			if capit > 0 {
				msg += fmt.Sprintf("- Capital letters found in domain: %d\n", capit)
			}
			if hexchar > 0 {
				msg += fmt.Sprintf("- Non Ascii characters found in domain: %d\n", capit)
			}
			if ent > SHANNON_THRESHOLD {
				msg += fmt.Sprintf("- Unexpected Shannon threshold: %f\n", ent)
			}
			if len(msg) > 0 {
				formattedEvent, _ := json.MarshalIndent(processEvent, "", " ")
				notifyChan <- fmt.Sprintf("Potential DNS tunelling for event:\n%s Vulnerabilities:\n%s", formattedEvent, msg)
			}
		}
	}
}
