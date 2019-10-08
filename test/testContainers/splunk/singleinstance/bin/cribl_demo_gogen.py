from __future__ import division
import sys
import os
import xml.dom.minidom
import subprocess
import logging
import logging.handlers
import platform
import urllib
import time


def setupLogger(logger=None, log_format='%(asctime)s %(levelname)s [Gogen] %(message)s', level=logging.INFO, log_name="gogen.log", logger_name="gogen"):
    """
    Setup a logger suitable for splunkd consumption
    """
    if logger is None:
        logger = logging.getLogger(logger_name)

    # Prevent the log messages from being duplicated in the python.log file
    logger.propagate = False
    logger.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(os.path.join(
        os.environ['SPLUNK_HOME'], 'var', 'log', 'splunk', log_name), maxBytes=2500000, backupCount=5)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)

    logger.debug("init %s logger", logger_name)
    return logger


SCHEME = """<scheme>
    <title>cribldemo Gogen</title>
    <description>Generate data using cribldemo Gogen</description>
    <use_external_validation>true</use_external_validation>
    <use_single_instance>false</use_single_instance>
    <streaming_mode>xml</streaming_mode>
    <endpoint>
        <args>
            <arg name="name">
                <title>GoGen input name</title>
                <description>Name of this GoGen input</description>
            </arg>

            <arg name="config_type">
                <title>Configuration Descriptor Type</title>
                <description>The type of config defined in the Configuration Descriptor field.  Defaults to config_dir.</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="config">
                <title>Configuration Descriptor</title>
                <description>Short Gogen path (coccyx/weblog for example), full file path,local file in config directory, or URL pointing to YAML or JSON config.  Leave blank to use all configs in gogen_assets.</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>

            <arg name="count">
                <title>Count</title>
                <description>Count of events to generate every interval.  Overrides any amounts set in the Gogen config</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="gogen_interval">
                <title>Interval</title>
                <description>Generate events every interval seconds.  Overrides any interval set in the Gogen config</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="end_intervals">
                <title>End Intervals</title>
                <description>Generate events for endIntervals and stop.  Overrides any endInterval set in the Gogen config</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="begin">
                <title>Begin</title>
                <description>Start generating events at begin time.  Can use Splunk's relative time syntax or an absolute time.  Overrides any begin setting in the Gogen config</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="end">
                <title>End</title>
                <description>End generating events at end time.  Can use Splunk's relative time syntax or an absolute time.  Overrides any end setting in the Gogen config</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>
            <arg name="generator_threads">
                <title>Generator Threads</title>
                <description>Sets number of generator threads</description>
                <required_on_edit>false</required_on_edit>
                <required_on_create>false</required_on_create>
            </arg>

            </args>
    </endpoint>
</scheme>
"""


def do_validate():
    config = get_validation_config()
    # TODO
    # if error , print_validation_error & sys.exit(2)

# prints validation error data to be consumed by Splunk


def print_validation_error(s):
    print "<error><message>%s</message></error>" % encodeXMLText(s)


def encodeXMLText(text):
    text = text.replace("&", "&amp;")
    text = text.replace("\"", "&quot;")
    text = text.replace("'", "&apos;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def usage():
    print "usage: %s [--scheme|--validate-arguments]"
    logger.error("Incorrect Program Usage")
    sys.exit(2)


def do_scheme():
    print SCHEME


# read XML configuration passed from splunkd
def get_config():
    config = {}

    try:
        # read everything from stdin
        config_str = sys.stdin.read()
        logger.debug("Config Str: %s" % config_str)

        # parse the config XML
        doc = xml.dom.minidom.parseString(config_str)
        root = doc.documentElement
        server_host = str(root.getElementsByTagName(
            "server_host")[0].firstChild.data)
        if server_host:
            logger.debug("XML: Found server_host")
            config["server_host"] = server_host
        server_uri = str(root.getElementsByTagName(
            "server_uri")[0].firstChild.data)
        if server_uri:
            logger.debug("XML: Found server_uri")
            config["server_uri"] = server_uri
        session_key = str(root.getElementsByTagName(
            "session_key")[0].firstChild.data)
        if session_key:
            logger.debug("XML: Found session_key")
            config["session_key"] = session_key
        checkpoint_dir = str(root.getElementsByTagName(
            "checkpoint_dir")[0].firstChild.data)
        if checkpoint_dir:
            logger.debug("XML: Found checkpoint_dir")
            config["checkpoint_dir"] = checkpoint_dir
        conf_node = root.getElementsByTagName("configuration")[0]
        if conf_node:
            logger.debug("XML: found configuration")
            stanza = conf_node.getElementsByTagName("stanza")[0]
            if stanza:
                stanza_name = stanza.getAttribute("name")
                if stanza_name:
                    logger.debug("XML: found stanza " + stanza_name)
                    config["name"] = stanza_name

                    params = stanza.getElementsByTagName("param")
                    for param in params:
                        param_name = param.getAttribute("name")
                        logger.debug("XML: found param '%s'" % param_name)
                        if param_name and param.firstChild and \
                           param.firstChild.nodeType == param.firstChild.TEXT_NODE:
                            data = param.firstChild.data
                            config[param_name] = data
                            logger.debug("XML: '%s' -> '%s'" %
                                         (param_name, data))

        checkpnt_node = root.getElementsByTagName("checkpoint_dir")[0]
        if checkpnt_node and checkpnt_node.firstChild and \
           checkpnt_node.firstChild.nodeType == checkpnt_node.firstChild.TEXT_NODE:
            config["checkpoint_dir"] = checkpnt_node.firstChild.data

        if not config:
            raise Exception, "Invalid configuration received from Splunk."

        # just some validation: make sure these keys are present (required)
        # validate_conf(config, "name")
        # validate_conf(config, "key_id")
        # validate_conf(config, "secret_key")
        # validate_conf(config, "checkpoint_dir")
    except Exception, e:
        raise Exception, "Error getting Splunk configuration via STDIN: %s" % str(
            e)

    return config

# read XML configuration passed from splunkd, need to refactor to support
# single instance mode


def get_validation_config():
    val_data = {}

    # read everything from stdin
    val_str = sys.stdin.read()

    # parse the validation XML
    doc = xml.dom.minidom.parseString(val_str)
    root = doc.documentElement

    logger.debug("XML: found items")
    item_node = root.getElementsByTagName("item")[0]
    if item_node:
        logger.debug("XML: found item")

        name = item_node.getAttribute("name")
        val_data["stanza"] = name

        params_node = item_node.getElementsByTagName("param")
        for param in params_node:
            name = param.getAttribute("name")
            logger.debug("Found param %s" % name)
            if name and param.firstChild and \
               param.firstChild.nodeType == param.firstChild.TEXT_NODE:
                val_data[name] = param.firstChild.data

    return val_data


if __name__ == '__main__':
    logger = setupLogger(level=logging.DEBUG)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--scheme":
            do_scheme()
        elif sys.argv[1] == "--validate-arguments":
            do_validate()
        else:
            usage()
        sys.exit(0)
    else:
        config = get_config()

        if platform.system() == 'Linux':
            exefile = 'gogen_real'
            gogen_url = 'https://api.gogen.io/linux/gogen'
        elif platform.system() == 'Windows':
            exefile = 'gogen_real.exe'
            gogen_url = 'https://api.gogen.io/windows/gogen.exe'
        else:
            exefile = 'gogen_real'
            gogen_url = 'https://api.gogen.io/osx/gogen'

        # gogen_path = os.path.join(
        # os.environ['SPLUNK_HOME'], 'etc', 'apps', 'splunk_app_gogen', 'bin',
        # exefile)
        gogen_base_path = os.path.sep.join(
            os.path.realpath(__file__).split(os.path.sep)[0:-2])
        gogen_path = os.path.join(gogen_base_path, 'bin', exefile)
        for i in range(5):
            try:
                if not os.path.exists(gogen_path):
                    tmp_path = gogen_path + '.tmp'
                    urllib.urlretrieve(gogen_url, tmp_path)
                    os.rename(tmp_path, gogen_path)
                    os.chmod(gogen_path, 0755)
                break
            except Exception as e:
                logger.error(
                    'failed to dowload gogen, retry=%d, err=%s' % (i,  str(err)))
                time.sleep(i+1)
                pass

        args = []
        args.append(gogen_path)
        # args.append('-v')
        args.append('-ot')
        args.append('modinput')

        if 'config_type' in config:
            config_type = str(config['config_type'])
        else:
            config_type = 'config_dir'

        if config_type == 'config_dir':
            args.append('-cd')
            args.append(os.path.join(gogen_base_path, 'gogen_assets'))
        else:
            args.append('-sd')
            args.append(os.path.join(gogen_base_path,
                                     'gogen_assets', 'samples') + os.path.sep)
            if 'config' in config:
                args.append('-c')
                config_file = str(config['config'])
                if config_type == 'local_file':
                    args.append(os.path.join(
                        gogen_base_path, 'configs', config_file))
                else:
                    args.append(config_file)

        if 'generator_threads' in config:
            args.append('-g')
            args.append(str(config['generator_threads']))

        args.append('gen')

        if 'count' in config:
            args.append('-c')
            args.append(str(config['count']))
        if 'gogen_interval' in config:
            args.append('-i')
            args.append(str(config['gogen_interval']))
        if 'end_intervals' in config:
            args.append('-ei')
            args.append(str(config['end_intervals']))
        if 'begin' in config:
            args.append('-b')
            args.append(str(config['begin']))
        if 'end' in config:
            args.append('-e')
            args.append(str(config['end']))
        # if 'begin' not in config and 'end' not in config and 'end_intervals' not in config:
        #     args.append('-r')

        import pprint
        logger.debug('args: %s' % pprint.pformat(args))
        logger.debug('command: %s' % ' '.join(args))

        sys.stdout.write("<stream>\n")
        sys.stdout.flush()
        p = subprocess.Popen(args, cwd=gogen_base_path,
                             shell=False)
