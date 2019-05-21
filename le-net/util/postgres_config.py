from configparser import ConfigParser
import psycopg2

CONFIG_FILE = "./database.ini"

def config(filename='database.ini', section='postgresql'):
  # create a parser
  parser = ConfigParser()
  # read config file
  parser.read(filename)

  # get section, default to postgresql
  db = {}
  if parser.has_section(section):
      params = parser.items(section)
      for param in params:
          db[param[0]] = param[1]
  else:
      raise Exception('Section {0} not found in the {1} file'.format(section, filename))

  return db

def connect(logger):
  """ Connect to the PostgreSQL database server """
  conn = None
  try:
    # read connection parameters
    params = config(CONFIG_FILE)

    # connect to the PostgreSQL server
    logger.info('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params)
    conn.autocommit=True

    # create a cursor
    cur = conn.cursor()

    logger.info("Connection created successfully")

    # close the communication with the PostgreSQL
    #cur.close()
    return cur
  except (Exception, psycopg2.DatabaseError) as error:
    logger.error("Connection failed")
  # finally:
  #   if conn is not None:
  #     conn.close()
  #     print('Database connection closed.')
