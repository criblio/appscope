use strict;
use warnings;
use LWP::UserAgent;

my $B = new LWP::UserAgent (agent => 'Mozilla/5.0', cookie_jar =>{}, );
$B->ssl_opts(verify_hostname => 0);
$B->ssl_opts(SSL_verify_mode => 0x00);

my $GET = $B->get('https://localhost')->content;
print $GET;