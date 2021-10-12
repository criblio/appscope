[cribl_demo_gogen://<name>]

config = <string>
  * Short Gogen path (coccyx/weblog for example), full file path,local file in config directory, or URL pointing to YAML or JSON config.

config_type  = <string>
  * local_file / short_path / full_file_path / url

count = <integer>
  * Count of events to generate every interval.  Overrides any amounts set in the Gogen config.

gogen_interval = <integer>
  * Generate events every interval seconds.  Overrides any interval set in the Gogen config.

end_intervals = <integer>
  * Generate events for endIntervals and stop.  Overrides any endInterval set in the Gogen config.

begin = <string>
  * Start generating events at begin time.  Can use Splunk's relative time syntax or an absolute time.  Overrides any begin setting in the Gogen config.

end = <string>
  * End generating events at end time.  Can use Splunk's relative time syntax or an absolute time.  Overrides any end setting in the Gogen config.

generator_threads = <integer>
  * Sets number of generator threads
