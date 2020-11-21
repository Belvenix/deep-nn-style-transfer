import configparser

CONFIG_FILE = 'config.ini'


def readConfig(configFile: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    try:
        with open(configFile) as f:
            config.read(configFile)
            return config
    except FileNotFoundError:
        raise Exception("Please set the working directory to the root of the project first!")


DNNConfigurer = readConfig(CONFIG_FILE)
